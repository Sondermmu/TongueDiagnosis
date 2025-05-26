import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_predictions(images, predictions, masks, save_path, num_samples=4):
    """可视化分割结果"""
    # 创建图像网格
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    # 反归一化函数
    def denormalize(x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = x.cpu().numpy().transpose(1, 2, 0)
        x = std * x + mean
        x = np.clip(x, 0, 1)
        return x

    for idx in range(min(num_samples, len(images))):
        # 显示原图
        axes[idx, 0].imshow(denormalize(images[idx]))
        axes[idx, 0].set_title('origin')
        axes[idx, 0].axis('off')

        # 显示预测mask
        axes[idx, 1].imshow(predictions[idx].cpu().numpy(), cmap='gray')
        axes[idx, 1].set_title('prediction')
        axes[idx, 1].axis('off')

        # 显示真实mask
        axes[idx, 2].imshow(masks[idx].cpu().numpy(), cmap='gray')
        axes[idx, 2].set_title('true')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_losses(train_losses, val_losses, save_path):
    """绘制损失曲线"""
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='eval')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.close()
    print(f"损失曲线已保存到 {save_path}")


def plot_cls_probabilities(class_names, probabilities, pred_class, save_path=None):
    """绘制分类概率图"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, probabilities, color='#4a86e8')

    # 高亮最高概率的条形
    bars[pred_class].set_color('#ff6b6b')

    plt.title('各类别预测概率', fontsize=16)
    plt.ylim(0, 1.0)

    # 在条形上方显示概率值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{probabilities[i]:.2f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=30, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"分类概率图已保存到 {save_path}")
        return None
    else:
        return plt.gcf()