import torch
import torch.nn as nn
from torchvision import models


def create_cls_model(model_name, num_classes, pretrained=True, device=None):
    """创建分类模型"""
    if model_name == 'ResNet':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'EfficientNet':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'ViT':
        model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    if device:
        model = model.to(device)

    return model


def load_cls_model(model_path, model_name, num_classes, device):
    """加载分类模型"""
    model = create_cls_model(model_name, num_classes, pretrained=False, device=device)
    print(f"分类模型加载成功，模型路径{model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model