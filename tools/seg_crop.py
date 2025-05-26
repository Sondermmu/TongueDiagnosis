import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.models.seg_model import load_seg_model
from src.datasets.seg_dataset import get_inference_transform
from config import SEG_CONFIG

def segment_and_crop(model, image_path, transform, device):
    image = np.array(Image.open(image_path).convert('RGB'))
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = (output.sigmoid() > 0.5).float() if output.shape[1] == 1 else torch.argmax(output, dim=1)
        mask = pred[0].cpu().numpy()
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.resize(image.shape[1::-1], Image.NEAREST)
    image = Image.fromarray(image)
    mask = np.array(mask)
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return np.array(image)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_image = np.array(image)[y_min:y_max+1, x_min:x_max+1]
    return cropped_image

def process_ori_data(ori_root, seg_root, model, transform, device):
    for cls in os.listdir(ori_root):
        cls_dir = os.path.join(ori_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        save_cls_dir = os.path.join(seg_root, cls)
        os.makedirs(save_cls_dir, exist_ok=True)
        for img_file in tqdm(os.listdir(cls_dir), desc=f"处理类别: {cls}"):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_dir, img_file)
                cropped = segment_and_crop(model, img_path, transform, device)
                save_name = os.path.splitext(img_file)[0] + "_seg.jpg"
                save_path = os.path.join(save_cls_dir, save_name)
                Image.fromarray(cropped).save(save_path)

if __name__ == "__main__":
    ori_root = "../data/ori_data"
    seg_root = "../data/seg_crop_data"
    device = SEG_CONFIG["DEVICE"]
    model_path = SEG_CONFIG["MODEL_PATH"]
    image_size = SEG_CONFIG["IMAGE_SIZE"]
    model = load_seg_model(model_path, device)
    transform = get_inference_transform(image_size)
    process_ori_data(ori_root, seg_root, model, transform, device)
    print("所有图片分割裁剪完成，结果保存在", seg_root)