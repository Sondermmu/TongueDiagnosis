import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.ce(y_pred, y_true)

def create_seg_model(encoder_name, encoder_weights, in_channels, classes, device):
    """创建分割模型"""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    ).to(device)

    return model


def load_seg_model(model_path, device):
    """加载分割模型"""
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.eval()
        print("分割模型加载成功")
        return model
    except Exception as e:
        print(f"分割模型加载失败: {e}")
        return None