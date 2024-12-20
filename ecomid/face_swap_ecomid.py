import cv2
import torch
import numpy as np
from PIL import Image
from pulid import attention_processor as attention
from pipeline_pulid_controlnet_86 import EcomIDPipeline
# from pipeline_pulid_controlnet import EcomIDPipeline
from insightface.app import FaceAnalysis
from transformers import pipeline
import facer
import os
# Disable GPU visibility. Make sure its BEFORE importing torch (or any other module that uses torch)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
device = torch.device(f'cuda:{0}')

# face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
insightface_app = FaceAnalysis(name='antelopev2', root='/data/wangqiqi/wangqiqi/ComfyUI/models/insightface/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
insightface_app.prepare(ctx_id=0, det_size=(640, 640))


def pred_face_info(img):
    face_info = insightface_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))#PIL图片对象一般为是RGBA或者RGB,而OpenCV通常使用BGR格式
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # only use the maximum face；这行代码的作用是根据每个面部的边界框的面积对面部信息进行排序
    print(face_info)
    

    return face_info # 函数返回面积最大的面部信息，这个信息包括了该面部的边界框坐标、面部特征点等数据。


def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in image_extensions]
    return image_files

bg_remove_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device='cuda')
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


import os
import random

random_seed = 0
generator = torch.Generator(device="cuda")
generator.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if __name__ == "__main__":

    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='/data/wangqiqi/wangqiqi/ComfyUI/models/insightface/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'/data/wangqiqi/wangqiqi/ComfyUI/models/instantid/ip-adapter.bin'
    controlnet_path = f'/data/wangqiqi/wangqiqi/face_swap_ecomid_v2/ControlNetModel'
    # controlnet_depth_path = f'diffusers/controlnet-depth-sdxl-1.0-small'
    pipeline = EcomIDPipeline(device)
    # other params
    DEFAULT_NEGATIVE_PROMPT = (
        'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
        'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
        'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
        'deformed, deformed eyeballs, cross-eyed,blurry'
    )

    attention.NUM_ZERO = 8
    attention.ORTHO = False
    attention.ORTHO_v2 = True


    # Infer setting
    # prompt = "realistic, high-quality photo of face, look forward, very detailed, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance"
    prompt = "a woman,light face,soft light,detailed"
    #prompt2 = "a woman,light face,soft light"
    #prompt1 = "a woman"
    # #"high-quality photo of a female face, eyes look forward, strong sunlight, very detailed, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance."
    #prompt = "a man "
    #n_prompt = 'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality'
    # n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), open mouth, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
    n_prompt = "(bangs:1.5), (fringe:1.5), (forehead hair:1.5), (uneven lighting:1.5), (inconsistent lighting:1.5), (freckles:1.5), (harsh shadows:1.5), (overexposed highlights:1.5), (underexposed shadows:1.5), (unnatural skin tones:1.5), patchy skin color, discolored skin, overly saturated skin, greenish skin tone, jaundiced appearance, sickly complexion, unrealistic skin texture, plastic-looking skin, extreme contrast, color imbalance, color banding, poor color grading, unflattering shadows on face"
    #n_prompt="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

    #  ./examples/poses/template_male_fixface2_croped_face.png    ./examples/poses/template_male_fixface2.jpg   16_gen.png  template_male_2_croped_face
    template_image_path = "/data/wangqiqi/wangqiqi/ComfyUI/muban_photo/2_croped_face.png"    # template_male_fixface2_croped_face.png template_male_end_test1_croped_face
    template_image_mask_path = "/data/wangqiqi/wangqiqi/fish/test_v0/template_cropped_face_mask.png" # template_male_end_test1_croped_mask
    control_image_path = "/data/wangqiqi/wangqiqi/ComfyUI/muban_photo/2_croped_face.png"

    pose_image = Image.open(template_image_path).convert('RGB')
    pose_image = pose_image.resize((928, 1024))
    mask_image = Image.open(template_image_mask_path).convert('RGB')
    mask_image = mask_image.resize((928, 1024))
    control_image = Image.open(control_image_path).convert('RGB')
    control_image = control_image.resize((928, 1024))

    face_files = get_image_files('/data/wangqiqi/female_test')  # female_test  male_normal

    # id_scale = 0.8
    steps = 50#50
    # strength = 0.9#0.75
    # guidance_scale = 0
    start_noise = 300#700
    smile_scale = 1
    strength_list = [0.8] #[0.8]
    id_scale_list = [0.5]#[0.6, 0.8]
    # guidance_scale_list = [0, 0.6, 1.0, 1.2, 1.5]
    guidance_scale_list = [2]#[2]
    for guidance_scale in guidance_scale_list:
        for strength in strength_list:
            for id_scale in id_scale_list:
                output_dir = f"/data/wangqiqi/wangqiqi/result/swapface/ecomid_python/output/ecomid/wqq_prompt2_imagelatent0.2noise08_strength{strength}_idscale{id_scale}_guidance_scale{guidance_scale}_step{steps}"#imagelatent0.2noise08
                os.makedirs(output_dir, exist_ok=True)
                for i, face_file in enumerate(face_files):
                    face_image = Image.open(face_file).convert('RGB')
                    face_info = pred_face_info(face_image)
                    face_embed = np.array(face_info['embedding'])[None, ...]#增加一个新的维度：使得面部嵌入向量的形状从 (128,) 变为 (1, 128)。这可以方便后续的批量处理（即使当前只有一个面部，也把它作为一个批次）。

                    id_embeddings = pipeline.get_id_embedding(np.array(face_image))
                    print(id_embeddings.shape)
                    print(id_embeddings)
                    
                    
                    torch.cuda.empty_cache()#---------------wqq
                    
                    img = pipeline.inference(prompt, (1, 1024, 928), control_image, face_embed, pose_image, mask_image,
                                             n_prompt, id_embeddings, id_scale, guidance_scale, steps, strength,
                                             start_noise, smile_scale)[0]
                    torch.cuda.empty_cache()#----------------------------wqq

                    face_name = os.path.basename(face_file).split('.')[0]
                    img.save(os.path.join(output_dir, face_name + ".png"))