import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from pulid import attention_processor as attention
from pipeline_ecomid import EcomIDPipeline
from insightface.app import FaceAnalysis
from transformers import pipeline
import facer
import math
import PIL.Image
import argparse

def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def pred_face_mask(img, face_info):
    points = []
    rects = []
    scores = []
    image_ids = []
    # for i in range(len(detected_faces)):

    points.append(face_info['kps'])
    rects.append(face_info['bbox'])
    scores.append(face_info['det_score'])
    image_ids.append(0)

    face_info = {}
    face_info['points'] = torch.tensor(points).to(device)
    face_info['rects'] = torch.tensor(rects).to(device)
    face_info['scores'] = torch.tensor(scores).to(device)
    face_info['image_ids'] = torch.tensor(image_ids).to(device)

    img = np.array(img)
    reordered_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device=device)

    with torch.inference_mode():
        faces = face_parser(reordered_img, face_info)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
    vis_img = vis_seg_probs.sum(0, keepdim=True)

    face_mask = vis_img != 0

    bbox = face_info['rects'][0]

    face_mask = face_mask.unsqueeze(0)
    face_mask = face_mask.squeeze().int()*255
    face_mask = face_mask.cpu().numpy()

    return face_mask


def prepareMaskAndPoseAndControlImage(pose_image, face_info):
    kps = face_info['kps']

    mask = np.zeros([height, width, 3])
    face_mask = pred_face_mask(pose_image, face_info)
    mask[face_mask>0] = 255
    face_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    face_mask = cv2.dilate(face_mask, kernel, iterations=2)
    face_mask = Image.fromarray(face_mask.astype(np.uint8))
    control_image = draw_kps(pose_image, kps)

    # (mask, pose, control PIL images), (original positon face + padding: x, y, w, h)
    return face_mask, control_image


def pred_face_info(img):
    face_info = insightface_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # only use the maximum face

    return face_info

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
    
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', required=True, help='Path for source image')
    parser.add_argument('--dst', required=True, help='Path for target image')
    parser.add_argument('--outdir', required=True, help='Path for storing output images')
    parser.add_argument('--face_adapter_path', required=True, help='Path for ipadpter')
    parser.add_argument('--controlnet_path', required=True, help='Path for controlnet model path')
    parser.add_argument('--base_model_path', required=True, help='Path for sdxl model path')
    parser.add_argument('--device', default="cuda:0")
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{0}')
    face_parser = facer.face_parser('farl/lapa/448', device=device)  # optional "farl/celebm/448"
    insightface_app = FaceAnalysis(name='antelopev2', root='./',
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    bg_remove_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device='cuda')
    pipeline_swap = EcomIDPipeline(args, insightface_app)
    
    # other params
    attention.NUM_ZERO = 8
    attention.ORTHO = False
    attention.ORTHO_v2 = True


    # Infer setting
    prompt = "a man "
    n_prompt = "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality"


    pose_image = Image.open(args.src).convert('RGB')
    width, height = pose_image.size

    face_info = insightface_app.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face

    mask_image, control_image = prepareMaskAndPoseAndControlImage(
        pose_image,
        face_info,
    )

    id_scale = 0.5
    steps = 30
    strength = 0.9
    guidance_scale = 3.5

    os.makedirs(args.outdir, exist_ok=True)
    face_image = Image.open(args.dst).convert('RGB')
    face_info = pred_face_info(face_image)
    face_embed = np.array(face_info['embedding'])[None, ...]

    id_embeddings = pipeline_swap.get_id_embedding(np.array(face_image))

    image = pipeline_swap.inference(prompt, (1, height, width), control_image, face_embed, pose_image, mask_image,
                             n_prompt, id_embeddings, id_scale, guidance_scale, steps, strength)[0]

    face_name = os.path.basename(args.dst).split('.')[0]
    image.save(os.path.join(args.outdir, face_name + f"_swaped.png"))
