import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import pandas as pd
import sys
from torch.utils.data import WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CLS_CONFIG
from src.datasets.cls_dataset import TongueClassificationDataset, get_cls_transforms
from src.models.cls_model import create_cls_model
from src.utils.metrics import calculate_cls_metrics
from src.utils.visualization import plot_losses


def train_cls_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name):
    """训练分类模型"""
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    precisions = []
    recalls = []
    f1_scores = []

    # 为当前模型创建保存路径
    model_save_dir = os.path.join(CLS_CONFIG["MODEL_PATH"], model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f"{model_name}_best.pth")
    final_model_path = os.path.join(model_save_dir, f"{model_name}.pth")

    counter = 0


    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training {model_name}"):
            inputs, labels = inputs.to(CLS_CONFIG["DEVICE"]), labels.to(CLS_CONFIG["DEVICE"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation {model_name}"):
                inputs, labels = inputs.to(CLS_CONFIG["DEVICE"]), labels.to(CLS_CONFIG["DEVICE"])

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        # 计算其他指标
        metrics = calculate_cls_metrics(all_preds, all_labels)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 记录日志
        log_msg = (f'Epoch [{epoch + 1}/{num_epochs}] - {model_name} - '
                   f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
                   f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, '
                   f'Precision: {metrics["precision"]:.4f}, '
                   f'Recall: {metrics["recall"]:.4f}, '
                   f'F1: {metrics["f1"]:.4f}')
        print(log_msg)
        logging.info(log_msg)

        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"保存最佳模型 {model_name}，验证准确率: {best_acc:.4f}")
            counter = 0
        else:
            counter += 1

        # 早停
        if counter >= CLS_CONFIG["PATIENCE"]:
            print("Early stopping")
            logging.info("Early stopping")
            break

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"保存最终模型 {model_name} 到 {final_model_path}")

    # 绘制损失曲线
    plot_losses(train_losses, val_losses, os.path.join(CLS_CONFIG["PLOTS_DIR"], f"{model_name}_loss.png"))


    return {
        'model_name': model_name,
        'best_acc': best_acc,
        'final_precision': precisions[-1],
        'final_recall': recalls[-1],
        'final_f1': f1_scores[-1],
        'best_model_path': best_model_path
    }

def compare_models(results):
    """比较不同模型的性能并可视化"""
    # 创建比较结果目录
    compare_dir = os.path.join(CLS_CONFIG["PLOTS_DIR"])
    os.makedirs(compare_dir, exist_ok=True)

    # 提取数据
    model_names = [r['model_name'] for r in results]
    accuracies = [r['best_acc'] for r in results]
    precisions = [r['final_precision'] for r in results]
    recalls = [r['final_recall'] for r in results]
    f1_scores = [r['final_f1'] for r in results]

    # 创建DataFrame
    df = pd.DataFrame({
        'model': model_names,
        'Acc': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # 绘制条形图比较
    metrics = ['Acc', 'Precision', 'Recall', 'F1 Score']

    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df[metric], width, label=metric)

    plt.xlabel('model')
    plt.ylabel('score')
    plt.xticks(x + width * 1.5, model_names)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "model_comparison.png"))
    plt.close()

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs(CLS_CONFIG["PLOTS_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(CLS_CONFIG["LOG_FILE"]), exist_ok=True)

    # 设置日志
    logging.basicConfig(filename=CLS_CONFIG["LOG_FILE"], level=logging.INFO,
                        format='%(message)s')

    # 设置随机种子
    np.random.seed(CLS_CONFIG["SEED"])
    torch.manual_seed(CLS_CONFIG["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CLS_CONFIG["SEED"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 获取数据路径
    train_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "data", "cls_data", "train")
    val_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data", "cls_data", "test")

    # 获取数据转换
    train_transform, val_transform = get_cls_transforms(CLS_CONFIG["IMAGE_SIZE"])

    # 类别名称
    CLASS_NAMES = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print("检测到的类别:", CLASS_NAMES)

    # 创建数据集
    train_dataset = TongueClassificationDataset(train_dir, transform=train_transform)
    val_dataset = TongueClassificationDataset(val_dir, transform=val_transform)

    # 计算每个样本的权重
    targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_sample_count = np.array([len(np.where(np.array(targets) == t)[0]) for t in range(len(CLASS_NAMES))])
    max_count = np.max(class_sample_count)
    weights = np.sqrt(max_count / class_sample_count)  # 使用平方根
    samples_weight = np.array([weights[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLS_CONFIG["BATCH_SIZE"],
        sampler=sampler,
        num_workers=CLS_CONFIG["NUM_WORKERS"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CLS_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CLS_CONFIG["NUM_WORKERS"]
    )

    # 定义要测试的模型列表
    model_configs = [
        'ResNet',
        'EfficientNet',
        'ViT'
    ]

    # 存储所有模型的结果
    all_results = []

    # 训练每个模型
    for model_name in model_configs:
        print(f"\n开始训练 {model_name}")
        logging.info(f"\n开始训练 {model_name}")

        # 创建模型
        model = create_cls_model(model_name, len(CLASS_NAMES), pretrained=True, device=CLS_CONFIG["DEVICE"])

        # 统计类别样本数
        label_counts = [0] * len(CLASS_NAMES)
        for cls_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(train_dir, cls)
            if os.path.exists(cls_dir):
                label_counts[cls_idx] = len(os.listdir(cls_dir))

        # 计算损失函数权重
        max_count = max(label_counts)
        class_weights = np.sqrt(max_count / np.array(label_counts))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(CLS_CONFIG["DEVICE"])

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=CLS_CONFIG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # 训练模型
        result = train_cls_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            CLS_CONFIG["EPOCHS"], model_name
        )

        all_results.append(result)

    # 比较所有模型的性能
    compare_models(all_results)



if __name__ == "__main__":
    train_dir = r'../../data/cls_data/train'
    print(sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]))
    main()