import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TongueSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, num_classes=2):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask[mask == 255] = 1

        if np.max(mask) >= self.num_classes:
            raise ValueError(f"Invalid mask value {np.max(mask)} for class {self.num_classes}.")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask

def get_seg_transforms(image_size, is_train=True):
    """获取分割模型的数据增强转换"""
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),  # 调整图像的亮度、对比度、饱和度和色调
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # 随机调整图像的伽马值
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.4),  # 应用仿射变换，包括缩放、平移和旋转
            A.GridDropout(ratio=0.1, p=0.2),  # 随机网格丢弃
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化图像
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),  # 调整图像大小
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化图像
            ToTensorV2(),  # 将图像转换为张量
        ])

def get_inference_transform(image_size):
    """获取推理时的转换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])