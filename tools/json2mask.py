import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw

def json_to_mask(json_path, mask_save_path, mask_size=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_path = data['imagePath']
    if mask_size is None:
        img = Image.open(os.path.join(os.path.dirname(json_path), img_path))
        w, h = img.size
    else:
        w, h = mask_size
    mask = Image.new('L', (w, h), 0)
    for shape in data['shapes']:
        points = shape['points']
        polygon = [tuple(point) for point in points]
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask) * 255
    Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)

def organize_seg_data(root_dir, images_dir, masks_dir):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.endswith('.json'):
                json_path = os.path.join(folder_path, fname)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                img_name = data['imagePath']
                img_path = os.path.join(folder_path, img_name)
                # 统一命名：类别名_原文件名
                base_name = f"{folder}_{os.path.splitext(img_name)[0]}"
                img_save_name = base_name + os.path.splitext(img_name)[1]
                mask_save_name = base_name + '_mask.png'
                img_save_path = os.path.join(images_dir, img_save_name)
                mask_save_path = os.path.join(masks_dir, mask_save_name)
                # 复制原图
                if os.path.exists(img_path):
                    shutil.copy2(img_path, img_save_path)
                # 生成mask
                json_to_mask(json_path, mask_save_path)
    print("完成")

if __name__ == "__main__":
    root_dir = r'..\data\ori_data'
    images_dir = r'..\data\seg_data\images'
    masks_dir = r'..\data\seg_data\annotations'
    organize_seg_data(root_dir, images_dir, masks_dir)