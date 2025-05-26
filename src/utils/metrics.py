import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_seg_metrics(preds, masks):
    """计算分割模型的评估指标"""
    preds_bool = (preds == 1)
    masks_bool = (masks == 1)

    # 计算各个区域的像素数
    intersection = (preds_bool & masks_bool).float().sum((1, 2))
    pred_area = preds_bool.float().sum((1, 2))
    mask_area = masks_bool.float().sum((1, 2))
    union = (preds_bool | masks_bool).float().sum((1, 2))

    # 计算各项指标
    dice = (2. * intersection + 1e-7) / (pred_area + mask_area + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)

    # 计算precision和recall
    tp = intersection
    fp = pred_area - intersection
    fn = mask_area - intersection

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_cls_metrics(all_preds, all_labels):
    """计算分类模型的评估指标"""
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }