import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import logging
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEG_CONFIG
from src.datasets.seg_dataset import TongueSegDataset, get_seg_transforms
from src.models.seg_model import create_seg_model, DiceCELoss
from src.utils.metrics import calculate_seg_metrics
from src.utils.visualization import plot_losses
from src.utils.visualization import visualize_predictions


def train_seg_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    """训练分割模型"""
    best_metrics = {'val_loss': float('inf'), 'val_dice': 0}
    train_losses = []
    val_losses = []
    metrics_history = []
    counter = 0

    for epoch in range(SEG_CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{SEG_CONFIG['EPOCHS']} - Training"):
            images, masks = images.to(SEG_CONFIG["DEVICE"]), masks.to(SEG_CONFIG["DEVICE"])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        epoch_metrics = {k: [] for k in ['dice', 'iou', 'precision', 'recall', 'f1']}  # 初始化指标字典

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{SEG_CONFIG['EPOCHS']} - Validation")):
                images, masks = images.to(SEG_CONFIG["DEVICE"]), masks.to(SEG_CONFIG["DEVICE"])
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                batch_metrics = calculate_seg_metrics(preds, masks)

                for k, v in batch_metrics.items():
                    epoch_metrics[k].extend(v.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 计算平均指标
        mean_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        metrics_history.append(mean_metrics)

        # 更新学习率
        scheduler.step(mean_metrics['dice'])  # 使用验证集的dice指标来调整学习率

        # 记录日志
        log_msg = (f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Dice: {mean_metrics['dice']:.4f}, "
                   f"Val IoU: {mean_metrics['iou']:.4f}, Val F1: {mean_metrics['f1']:.4f}")
        print(log_msg)
        logging.info(log_msg)


        # 根据dice指标保存最佳模型
        if mean_metrics['dice'] > best_metrics['val_dice']:
            best_metrics['val_dice'] = mean_metrics['dice']
            torch.save(model.state_dict(), SEG_CONFIG["BEST_MODEL_PATH"])
            counter = 0
        else:
            counter += 1

        if counter >= SEG_CONFIG["PATIENCE"]:
            print("Early stopping")
            logging.info("Early stopping")
            break

    return train_losses, val_losses, metrics_history


def evaluate_seg_model(model, dataloader):
    """评估分割模型"""
    model.eval()
    all_dice_scores = []
    all_iou_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images, masks = images.to(SEG_CONFIG["DEVICE"]), masks.to(SEG_CONFIG["DEVICE"])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 只考虑前景区域
            foreground = (masks == 1)
            preds_fg = (preds == 1) & foreground
            masks_fg = foreground

            intersection = (preds_fg & masks_fg).float().sum((1, 2))
            pred_area = preds_fg.float().sum((1, 2))
            mask_area = masks_fg.float().sum((1, 2))
            union = ((preds == 1) | (masks == 1)).float().sum((1, 2))

            dice = (2. * intersection + 1e-7) / (pred_area + mask_area + 1e-7)
            iou = (intersection + 1e-7) / (union + 1e-7)
            tp = intersection
            fp = pred_area - intersection
            fn = mask_area - intersection
            precision = (tp + 1e-7) / (tp + fp + 1e-7)
            recall = (tp + 1e-7) / (tp + fn + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

            all_dice_scores.extend(dice.cpu().numpy())
            all_iou_scores.extend(iou.cpu().numpy())
            all_precision_scores.extend(precision.cpu().numpy())
            all_recall_scores.extend(recall.cpu().numpy())
            all_f1_scores.extend(f1.cpu().numpy())

            if batch_idx == 0:
                visualize_predictions(
                    images.cpu(), preds.cpu(), masks.cpu(),
                    save_path=os.path.join(SEG_CONFIG["PLOTS_DIR"], "分割效果.png"),
                    num_samples=min(4, images.size(0))
                )


    mean_dice = np.mean(all_dice_scores)
    mean_iou = np.mean(all_iou_scores)
    mean_precision = np.mean(all_precision_scores)
    mean_recall = np.mean(all_recall_scores)
    mean_f1 = np.mean(all_f1_scores)

    msg = (f"Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}, "
            f"Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, Mean F1: {mean_f1:.4f}")
    print(msg)
    logging.info(msg)

    return mean_dice, mean_iou, mean_precision, mean_recall, mean_f1


def run_kfold_cross_validation(k=5):
    # 创建必要的目录
    os.makedirs(SEG_CONFIG["PLOTS_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(SEG_CONFIG["LOG_FILE"]), exist_ok=True)

    # 设置日志
    logging.basicConfig(filename=SEG_CONFIG["LOG_FILE"], level=logging.INFO,
                        format='%(message)s')
    # 获取数据路径
    image_paths = sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "data", "seg_data", "images", "*.jpg")))
    mask_paths = sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "data", "seg_data", "annotations", "*.png")))

    image_paths = np.array(image_paths)
    mask_paths = np.array(mask_paths)

    image_files = sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "seg_data", "images", "*.*")))
    mask_files = sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "seg_data", "annotations", "*_mask.png")))
    
    # 建立图片名到路径的映射（不含扩展名）
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    mask_dict = {os.path.splitext(os.path.basename(p))[0].replace('_mask', ''): p for p in mask_files}
    
    # 有对应关系的图片和mask
    common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
    image_paths = np.array([image_dict[k] for k in common_keys])
    mask_paths = np.array([mask_dict[k] for k in common_keys])

    kf = KFold(n_splits=k, shuffle=True, random_state=SEG_CONFIG["SEED"])
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\n===== Fold {fold+1}/{k} =====")
        logging.info(f"\n===== Fold {fold+1}/{k} =====")
        
        # 划分数据
        train_imgs, train_masks = image_paths[train_idx], mask_paths[train_idx]
        val_imgs, val_masks = image_paths[val_idx], mask_paths[val_idx]

        # 创建数据集和dataloader
        train_transform = get_seg_transforms(SEG_CONFIG["IMAGE_SIZE"], is_train=True)
        val_transform = get_seg_transforms(SEG_CONFIG["IMAGE_SIZE"], is_train=False)
        train_ds = TongueSegDataset(train_imgs, train_masks, train_transform)
        val_ds = TongueSegDataset(val_imgs, val_masks, val_transform)
        train_loader = DataLoader(train_ds, batch_size=SEG_CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=SEG_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)

        # 创建模型
        model = create_seg_model(
            encoder_name=SEG_CONFIG["ENCODER"],
            encoder_weights=SEG_CONFIG["ENCODER_WEIGHTS"],
            in_channels=3,
            classes=SEG_CONFIG["CLASS"],
            device=SEG_CONFIG["DEVICE"],
        )

        # 损失函数、优化器、调度器
        criterion = DiceCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=SEG_CONFIG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

        # 训练
        train_losses, val_losses, metrics_history = train_seg_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )

        # 加载最佳模型权重
        model.load_state_dict(torch.load(SEG_CONFIG["BEST_MODEL_PATH"]))

        # 评估
        mean_dice, mean_iou, mean_precision, mean_recall, mean_f1 = evaluate_seg_model(model, val_loader)
        fold_metrics.append({
            "dice": mean_dice,
            "iou": mean_iou,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1
        })

        # 保存整个模型
        torch.save(model, SEG_CONFIG["MODEL_PATH"])

        plot_losses(train_losses, val_losses,
                    os.path.join(SEG_CONFIG["PLOTS_DIR"], f"loss_fold_{fold + 1}.png"))

    # 汇总结果
    print("\n===== K折交叉验证结果 =====")
    logging.info("\n===== K折交叉验证结果 =====")  
    for i, m in enumerate(fold_metrics):
        print(f"Fold {i+1}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, F1={m['f1']:.4f}")
        logging.info(f"Fold {i+1}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, F1={m['f1']:.4f}") 
    print("平均：")
    logging.info("平均：")  
    print("Dice: {:.4f}".format(np.mean([m['dice'] for m in fold_metrics])))
    logging.info("Dice: {:.4f}".format(np.mean([m['dice'] for m in fold_metrics])))  
    print("IoU: {:.4f}".format(np.mean([m['iou'] for m in fold_metrics])))
    logging.info("IoU: {:.4f}".format(np.mean([m['iou'] for m in fold_metrics])))
    print("F1: {:.4f}".format(np.mean([m['f1'] for m in fold_metrics])))
    logging.info("F1: {:.4f}".format(np.mean([m['f1'] for m in fold_metrics]))) 

if __name__ == "__main__":
    run_kfold_cross_validation(k=5)