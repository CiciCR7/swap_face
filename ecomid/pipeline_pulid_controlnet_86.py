import gc
import numpy as np
import cv2
import insightface
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel
from pipeline_stable_diffusion_xl_inpaint_ecomid_86 import StableDiffusionXLInpaintPulidPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from diffusers.models import ControlNetModel
from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders import IDEncoder
from pulid.utils import img2tensor, is_torch2_available, tensor2img

if is_torch2_available():
    from pulid.attention_processor import AttnProcessor2_0 as AttnProcessor
    from pulid.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from pulid.attention_processor import AttnProcessor, IDAttnProcessor


class EcomIDPipeline:
    def __init__(self, device, *args, **kwargs):
        super().__init__()
        self.device = device

        # Path to InstantID models
        face_adapter_path = f'/data/wangqiqi/wangqiqi/ComfyUI/models/instantid/ip-adapter.bin'
        controlnet_path = f'/data/wangqiqi/wangqiqi/face_swap_ecomid_v2/ControlNetModel'
        # lora_weight = f'checkpoints/smiling.pt'
        # lora_weight = f'/data2/wangqiqi/ComfyUI/models/loras/gender_slider-sdxl.safetensors'
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

        # base_model_path = f'/data2/wangcairong/fishing_proj/face_swap/checkpoints/StableDiffusion/realisticStockPhoto_v20.safetensors'
        # base_model_path = f'/data2/wangqiqi/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors'
        base_model_path = f'/data/wangqiqi/wangqiqi/ComfyUI/models/checkpoints/realismEngineSDXL_v30VAE.safetensors'

        self.pipe = StableDiffusionXLInpaintPulidPipeline.from_single_file(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)


        self.hack_unet_attn_layers(self.pipe.unet)
        self.pipe.watermark = None


        # scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.load_ip_adapter_instantid(face_adapter_path)
        # ID adapters
        self.id_adapter = IDEncoder().to(self.device)

        # self.pipe.load_sliders(lora_weight, self.device)#-------wqq
        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='/data/wangqiqi/wangqiqi/ComfyUI/models/insightface/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()
        self.lonviad_pretrain()
        print("---pretrain完成")
        torch.cuda.empty_cache()   #----wqq

        # other configs
        self.debug_img_list = []

    def hack_unet_attn_layers(self, unet):#对 U-Net 网络中注意力层的处理，主要目的是根据不同的层类型为 U-Net 模型中的每一层设置适当的注意力处理器（Attention Processor），并将这些处理器保存在一个列表中。代码的关键部分涉及到如何动态地创建和管理不同的注意力处理器。
        id_adapter_attn_procs = {}#这行代码初始化一个空字典 id_adapter_attn_procs，用于存储每个注意力处理器对应的层名称。字典的键是注意力层的名称，值是相应的 IDAttnProcessor 或 AttnProcessor 实例。
        for name, _ in unet.attn_processors.items():#这行代码开始循环遍历 U-Net 中所有的注意力处理器。unet.attn_processors 是一个字典，包含了 U-Net 模型中的所有注意力层（通常是 Attention 层）及其相关处理器。每次循环中，name 是注意力层的名称，_ 是值，但在这个场景下，我们只关心名称。
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):#接下来的 if-elif 语句决定了每个注意力层的 hidden_size，也就是输入到注意力机制中的特征维度大小：
                hidden_size = unet.config.block_out_channels[-1]#取模型配置中最后一个块的输出通道数。#mid_block只有一个
            elif name.startswith("up_blocks"):#hidden_size 是用来表示每个块的输出维度，也就是每个块输出的特征的通道数。中间块是网络的核心部分，通常具有较大的输出维度，因为它处理的是整张图像的高层特征。
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]#根据 block_id 来确定该块的 hidden_size，它是 block_out_channels 列表中对应的通道数。这里通过 reversed() 反转了 block_out_channels，因为上采样块的输出维度通常是从高维到低维的。
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])#down_blocks.2
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)#将新创建的 id_adapter_attn_procs 字典（包含了为每个注意力层配置的注意力处理器）设置到 unet 模型中。set_attn_processor() 方法将更新 unet 模型的注意力处理器。
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())#将 unet.attn_processors 中的所有注意力处理器（即 IDAttnProcessor 和 AttnProcessor）的值存储到 self.id_adapter_attn_layers 中，并使用 nn.ModuleList 封装它。这是为了方便后续在模型训练或推理过程中管理和访问所有的注意力处理器。

    def lonviad_pretrain(self):
        # hf_hub_download('guozinan/PuLID', 'pulid_v1.bin', local_dir='models')
        ckpt_path = '/data/wangqiqi/wangqiqi/fish/face_swap-master/pulid_models/pulid_v1.bin'

        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]#  对齐的人脸，正的人脸
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)#numpy->tensor
        if id_ante_embedding.ndim == 1:#id_ante_embedding.ndim 返回张量的维度数#ndim == 1 表示 id_ante_embedding 是一个一维张量，通常是一个形如 (N,) 的向量，其中 N 是元素的个数。
            id_ante_embedding = id_ante_embedding.unsqueeze(0)
            #unsqueeze(0) 操作会在张量的第一个维度上增加一个新的维度。也就是说，它将原来的形状从 (N,) 转换为 (1, N)。

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0#参数 bgr2rgb=True 表示将图像从 BGR 色彩通道（OpenCV 常用格式）转换为 RGB（通常是深度学习模型的要求）。#.unsqueeze(0) 为原来形状为 (C, H, W) 的张量变为 (1, C, H, W)，其中 1 表示批量大小（Batch Size）。参数 align_face 是输入的对齐人脸图像（可能是已经经过对齐处理的面部图像）。
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        #对输入图像进行归一化操作。减去均值 [0.485, 0.456, 0.406] 并除以标准差 [0.229, 0.224, 0.225]，这是常见的归一化操作，用于调整图像值到模型的输入标准。调用 face_helper 的 face_parse 方法，这通常是一个人脸解析网络，用于对输入图像进行语义分割。输出 parsing_out 是一个分割结果，通常为 (1, num_classes, H, W) 的张量，其中 num_classes 是语义分割类别的数量。取出批次维度（即第一个图像的分割结果）。结果 parsing_out 形状变为 (num_classes, H, W)。
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        #沿着类别维度（dim=1）找到分割类别的最大值索引。返回的结果是每个像素的类别标签，形状变为 (1, H, W)。keepdim=True保留维度，确保结果仍是三维张量 (1, H, W)。
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]#定义背景类别标签的列表 bg_label。这些数字代表在语义分割中被视为背景的类别，例如头发、脖子或其他非脸部部分。
        bg = sum(parsing_out == i for i in bg_label).bool()#遍历 bg_label 中的每个标签 i，生成一个布尔张量，标记哪些像素属于背景。将所有布尔张量相加，得到一个张量，其中背景类别的像素值为 True，其他像素值为 False。

        white_image = torch.ones_like(input)#创建一个与 input 形状相同的张量 white_image，所有值为 1，通常表示白色图像（归一化的 RGB 值）。
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))#根据布尔张量 bg 的值，选择白色图像或灰度图像。
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))#张量转换为图像格式（例如 NumPy 数组）# 将生成的图像添加到调试图像列表 self.debug_img_list 中，用于后续检查或显示。

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)#使用双三次插值（BICUBIC）进行图像大小调整。双三次插值是一种平滑的图像插值方法，用于在图像放大或缩小时减少失真。
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)#self.eva_transform_mean 和 self.eva_transform_std 是用于归一化的均值和标准差，通常是 ImageNet 数据集的标准值 [0.485, 0.456, 0.406]（均值）和 [0.229, 0.224, 0.225]（标准差）
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image, return_all_features=False, return_hidden=True, shuffle=False
        )#  表示不返回所有的特征，而只返回最终的视觉特征。表示返回模型的隐藏层特征（即中间特征），而不仅仅是最终输出。不对输入进行洗牌，通常用于保持图像顺序。第一个变量存储模型对输入图像（face_features_image）的视觉编码结果，通常是一个表示图像的特征向量。第二个存储来自 ViT（Vision Transformer）模型的所有中间隐藏层的特征，可以用来进行更复杂的分析。
        print("====下面是id_cond_vit")
        print(id_cond_vit)
        print("====下面是id_vit_hidden")
        print(id_vit_hidden)#多个tensor代表多个隐藏层，id_vit_hidden 是一个包含多个元素的 列表，每个元素都是一个 tensor，这些 tensor 是从 ViT 模型中不同层的输出。
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)# id_cond_vit_norm 是 id_cond_vit 每个向量的模长，用于后续的归一化处理。计算 id_cond_vit 在维度 1（特征维度）上的 L2 范数（即每个特征向量的模长）。
# torch.norm(..., 2, 1) 计算 L2 范数，2 表示二范数，1 表示计算维度（特征维度）。
# True 表示保持维度输出，返回一个和输入形状相同的张量。
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)# 将 id_cond_vit 的每个特征向量除以它的 L2 范数 id_cond_vit_norm，实现对特征向量的归一化。结果是每个向量的长度被标准化为 1，这样特征向量具有单位长度。

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)# 沿着最后一个维度（dim=-1）连接。，一个tensor
        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):#遍历 id_vit_hidden 中的每一层隐藏特征。
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))#对于 id_vit_hidden 中的每一层，创建一个与该层相同形状的全零张量，并将其添加到 id_vit_hidden_uncond 列表中。

        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        # return id_embedding
        return torch.cat((uncond_id_embedding, id_embedding), dim=0)

    def inference(self, prompt, size, control_image, face_embed, init_image, mask_image, prompt_n='',  image_embedding=None, id_scale=1.0, guidance_scale=1.2, steps=30 ,strength= 0.99, start_noise=700,
            smile_scale=1):
        images = self.pipe(
            prompt=prompt,
            negative_prompt=prompt_n,
            num_images_per_prompt=size[0],
            height=size[1],
            width=size[2],
            image=[control_image],
            controlnet_conditioning_scale=[0.5],
            image_embeds=face_embed,
            image_init=init_image,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            start_noise=start_noise,
            scale=smile_scale,
            cross_attention_kwargs={'id_embedding': image_embedding, 'id_scale': id_scale},
        )[0]

        return images
