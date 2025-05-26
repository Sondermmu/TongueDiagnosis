import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(src_root, dst_root, test_size=0.2):
    for cls in os.listdir(src_root):
        cls_dir = os.path.join(src_root, cls)
        print(f"正在处理{cls_dir}")
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        train_imgs, test_imgs = train_test_split(imgs, test_size=test_size, random_state=42)
        for split, split_imgs in zip(['train', 'test'], [train_imgs, test_imgs]):
            split_dir = os.path.join(dst_root, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(os.path.join(cls_dir, img), os.path.join(split_dir, img))

if __name__ == "__main__":
    src_root = r'../data/seg_crop_data'
    dst_root = r'../data/cls_data'
    split_data(src_root, dst_root)
    print("图片已划分为训练集和测试集，结果保存在", dst_root)