import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import cv2
import os
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QSizePolicy, QProgressBar, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEG_CONFIG, CLS_CONFIG, CLASS_NAMES, TONGUE_ADVICE, TEMP_DIR
from src.models.seg_model import load_seg_model
from src.models.cls_model import load_cls_model
from src.datasets.cls_dataset import get_inference_transform
from src.datasets.seg_dataset import get_inference_transform as get_seg_inference_transform

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


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


class ProcessingThread(QThread):
    """处理图像的后台线程"""
    finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int)

    def __init__(self, parent, image_path, seg_model, class_model, transform, seg_transform, device):
        super().__init__(parent)
        self.image_path = image_path
        self.seg_model = seg_model
        self.class_model = class_model
        self.transform = transform
        self.seg_transform = seg_transform
        self.device = device

    def run(self):
        try:
            # 阶段1：读取图像
            self.progress_updated.emit(10)
            original_image = Image.open(self.image_path).convert("RGB")

            # 阶段2：分割图像
            self.progress_updated.emit(30)
            cropped_image = segment_and_crop(self.seg_model, self.image_path, self.seg_transform, self.device)
            segmented_pil = Image.fromarray(cropped_image)

            # 临时保存分割后的图像用于分类
            temp_path = os.path.join(TEMP_DIR, "temp_segmented.jpg")
            segmented_pil.save(temp_path)

            # 阶段3：分类预测
            self.progress_updated.emit(80)
            input_tensor = self.transform(segmented_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.class_model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                pred_prob, pred_class = torch.max(probabilities, 1)
                pred_class_name = CLASS_NAMES[pred_class.item()]

            self.progress_updated.emit(100)

            result = {
                'original_image': original_image,
                'segmented_image': segmented_pil,
                'probabilities': probabilities,
                'pred_class': pred_class.item(),
                'pred_class_name': pred_class_name,
                'pred_prob': pred_prob.item()
            }

            self.finished.emit(result)

        except Exception as e:
            print(f"处理图像时出错: {e}")
            traceback.print_exc()
            self.finished.emit(None)


class TongueAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('舌象分析系统')
        # 增大初始窗口尺寸
        self.setGeometry(100, 100, 2000, 1000)

        # 应用样式
        self.apply_styles()

        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载分割模型
        self.seg_model = load_seg_model(SEG_CONFIG["MODEL_PATH"], self.device)

        # 加载分类模型
        self.class_model = load_cls_model(
            os.path.join(CLS_CONFIG["MODEL_PATH"], 'ResNet', 'ResNet.pth'),
            'ResNet',
            len(CLASS_NAMES),
            self.device
        )

        # 获取转换函数
        self.transform = get_inference_transform(CLS_CONFIG["IMAGE_SIZE"])
        self.seg_transform = get_seg_inference_transform(SEG_CONFIG["IMAGE_SIZE"])

        # 保存当前图像和分割后的图像
        self.current_image = None
        self.segmented_image = None
        self.processing_thread = None

        # 创建临时目录
        os.makedirs(TEMP_DIR, exist_ok=True)

        self.init_ui()

    def apply_styles(self):
        """应用全局样式"""
        # 增大基础字体大小
        base_font_size = 24

        self.setStyleSheet(f"""
            QMainWindow, QLabel, QPushButton, QProgressBar {{
                font-family: 'Microsoft YaHei';
                font-size: {base_font_size}px;
            }}
            QLabel {{
                color: #333333;
            }}
            QPushButton {{
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: {base_font_size}px;
                border-radius: 6px;
                transition: all 0.3s;
            }}
            QPushButton:hover {{
                background-color: #3a76d8;
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                transform: translateY(1px);
            }}
            QFrame {{
                border-radius: 12px;
                background-color: white;
                border: 1px solid #e9ecef;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}
            QProgressBar {{
                border: 1px solid #e9ecef;
                border-radius: 6px;
                text-align: center;
                height: 24px;
                margin-top: 12px;
            }}
            QProgressBar::chunk {{
                background-color: #4a86e8;
                border-radius: 5px;
            }}
            .TitleLabel {{
                font-size: 28px;
                font-weight: bold;
            }}
            .SubtitleLabel {{
                font-size: 16px;
                opacity: 0.8;
            }}
            .ResultTitle {{
                font-size: 24px;
                font-weight: bold;
            }}
            .ResultLabel {{
                font-size: {base_font_size}px;
            }}
            .ResultValue {{
                font-size: 20px;
                font-weight: bold;
                color: #ff6b6b;
            }}
        """)

    def init_ui(self):
        """初始化用户界面"""
        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(24)
        main_layout.setContentsMargins(24, 24, 24, 24)

        # 左侧布局：图片显示和选择按钮
        left_frame = QFrame()
        left_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(24)

        # 标题区域
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #4a86e8; border-radius: 10px;")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(24, 24, 24, 24)

        title_label = QLabel('舌象分析系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("TitleLabel")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel('中医舌诊辅助分析工具')
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        subtitle_label.setObjectName("SubtitleLabel")
        title_layout.addWidget(subtitle_label)

        left_layout.addWidget(title_frame)

        # 图像选择区域
        select_frame = QFrame()
        select_layout = QVBoxLayout(select_frame)
        select_layout.setContentsMargins(24, 24, 24, 24)

        select_label = QLabel('选择舌象图像')
        select_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 12px;")
        select_layout.addWidget(select_label)

        select_button = QPushButton('浏览文件')
        select_button.setFixedHeight(48)
        select_button.setIcon(QIcon.fromTheme('document-open'))
        select_button.clicked.connect(self.load_image)
        select_layout.addWidget(select_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()  # 初始隐藏进度条
        self.progress_bar.setTextVisible(True)
        select_layout.addWidget(self.progress_bar)

        left_layout.addWidget(select_frame)

        # 图像显示区域 - 增大图像区域大小
        image_group = QFrame()
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(24, 24, 24, 24)

        # 原始图像显示
        original_frame = QFrame()
        original_frame.setStyleSheet("background-color: #f9f9f9; border-radius: 10px;")
        original_layout = QVBoxLayout(original_frame)
        original_layout.setContentsMargins(18, 18, 18, 18)

        original_title = QLabel('原始舌象图像')
        original_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 12px;")
        original_layout.addWidget(original_title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(350)  # 增大最小高度
        self.image_label.setStyleSheet("border: 2px dashed #e9ecef; background-color: #ffffff; border-radius: 8px;")

        original_placeholder = QLabel("请选择一张舌象图像")
        original_placeholder.setAlignment(Qt.AlignCenter)
        original_placeholder.setStyleSheet("color: #888888; font-size: 22px;")

        original_layout.addWidget(self.image_label)
        original_layout.addWidget(original_placeholder)

        image_layout.addWidget(original_frame)

        # 分割后图像显示
        segmented_frame = QFrame()
        segmented_frame.setStyleSheet("background-color: #f9f9f9; border-radius: 10px;")
        segmented_layout = QVBoxLayout(segmented_frame)
        segmented_layout.setContentsMargins(18, 18, 18, 18)

        segmented_title = QLabel('分割后的舌象图像')
        segmented_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 12px;")
        segmented_layout.addWidget(segmented_title)

        self.seg_image_label = QLabel()
        self.seg_image_label.setAlignment(Qt.AlignCenter)
        self.seg_image_label.setMinimumHeight(350)
        self.seg_image_label.setStyleSheet("border: 2px dashed #e9ecef; background-color: #ffffff; border-radius: 8px;")

        segmented_placeholder = QLabel("分割后的舌象将显示在这里")
        segmented_placeholder.setAlignment(Qt.AlignCenter)
        segmented_placeholder.setStyleSheet("color: #888888; font-size: 22px;")

        segmented_layout.addWidget(self.seg_image_label)
        segmented_layout.addWidget(segmented_placeholder)

        image_layout.addWidget(segmented_frame)

        left_layout.addWidget(image_group)

        # 右侧布局：预测结果和建议
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(24)

        # 结果标题
        result_title = QLabel('分析结果')
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 26px; font-weight: bold;")
        result_title.setObjectName("ResultTitle")
        right_layout.addWidget(result_title)

        # 预测概率图
        chart_frame = QFrame()
        chart_frame.setMinimumHeight(350)  # 最小高度
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(24, 24, 24, 24)

        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: white; border-radius: 10px;")

        chart_layout.addWidget(self.canvas)
        right_layout.addWidget(chart_frame)

        # 使用滚动区域来显示预测结果和建议
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        result_scroll.setStyleSheet("border: none;")

        result_content = QWidget()
        result_layout = QVBoxLayout(result_content)
        result_layout.setContentsMargins(0, 0, 0, 0)

        # 预测结果和建议
        result_frame = QFrame()
        result_frame.setStyleSheet("background-color: #f0f7ff; border-radius: 10px; padding: 18px;")
        result_inner_layout = QVBoxLayout(result_frame)

        self.result_label = QLabel('预测结果：')
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setWordWrap(True)

        self.desc_label = QLabel('描述：')
        self.desc_label.setObjectName("ResultLabel")
        self.desc_label.setWordWrap(True)

        self.advice_label = QLabel('建议：')
        self.advice_label.setObjectName("ResultLabel")
        self.advice_label.setWordWrap(True)

        for label in [self.result_label, self.desc_label, self.advice_label]:
            label.setMinimumHeight(60)
            result_inner_layout.addWidget(label)

        # 让result_frame自动填满剩余空间
        result_layout.addWidget(result_frame, 1)
        # 去掉addStretch
        # result_layout.addStretch(2)  # 删除或注释掉这一行

        result_scroll.setWidget(result_content)
        right_layout.addWidget(result_scroll, 1)  # 让结果区域占据剩余空间

        # 调整左右布局比例
        main_layout.addWidget(left_frame, 4)  # 左侧占4份
        main_layout.addWidget(right_frame, 6)  # 右侧占6份

        # 初始化界面状态
        self.reset_ui_state()

    def reset_ui_state(self):
        """重置UI状态到初始状态"""
        # 清空图像显示区域
        self.image_label.clear()
        self.seg_image_label.clear()

        # 清空结果区域
        self.result_label.setText("预测结果：")
        self.desc_label.setText("描述：")
        self.advice_label.setText("建议：")

        # 清空图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title('各类别预测概率', fontsize=18)
        ax.set_ylim(0, 1.0)
        ax.text(0.5, 0.5, '请选择一张舌象图像进行分析',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=20, color='#888888')
        self.canvas.draw()

    def load_image(self):
        """加载图像并开始处理"""
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files (*.jpg *.png)')
        if file_name:
            # 显示原始图像
            self.current_image = Image.open(file_name).convert("RGB")
            self.display_image(self.image_label, file_name)

            # 重置分割图像显示
            self.seg_image_label.clear()

            # 显示进度条并开始处理
            self.progress_bar.setValue(0)
            self.progress_bar.show()

            # 创建处理线程
            self.processing_thread = ProcessingThread(
                self, file_name, self.seg_model, self.class_model,
                self.transform, self.seg_transform, self.device
            )
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.start()

    def display_image(self, label, image_path):
        """在标签上显示图像"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def on_processing_finished(self, result):
        """处理完成后的回调函数"""
        # 隐藏进度条
        self.progress_bar.hide()

        if result:
            # 保存分割后的图像
            self.segmented_image = result['segmented_image']

            # 显示分割后的图像
            temp_path = os.path.join(TEMP_DIR, "temp_segmented.jpg")
            self.segmented_image.save(temp_path)
            self.display_image(self.seg_image_label, temp_path)

            # 更新预测概率图
            self.update_probability_chart(result)

            # 更新分析结果
            self.update_analysis_results(result)
        else:
            # 处理失败
            self.seg_image_label.setText("图像分割失败")

    def update_probability_chart(self, result):
        """更新概率分布图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        probs = result['probabilities'][0].cpu().numpy()

        # 使用水平条形图
        bars = ax.barh(CLASS_NAMES, probs, color='#4a86e8', alpha=0.8)

        # 高亮最高概率的条形
        bars[result['pred_class']].set_color('#ff6b6b')

        ax.set_title('各类别预测概率', fontsize=20)
        ax.set_xlim(0, 1.0)

        # 在条形旁边显示概率值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{probs[i]:.2f}', ha='left', va='center', fontsize=16)

        plt.yticks(fontsize=14)
        plt.tight_layout()
        self.canvas.draw()

    def update_analysis_results(self, result):
        """更新分析结果"""
        pred_class_name = result['pred_class_name']
        pred_prob = result['pred_prob']

        # 使用统一的样式类
        self.result_label.setText(
            f'<b>预测结果：</b> <span class="ResultValue">{pred_class_name}</span> (置信度: {pred_prob:.2%})')

        if pred_class_name in TONGUE_ADVICE:
            self.desc_label.setText(f'<b>舌象：</b> {TONGUE_ADVICE[pred_class_name]["舌象"]}')

            # 格式化建议文本
            advice_text = TONGUE_ADVICE[pred_class_name]["建议"]
            advice_html = "<b>建议：</b><br>" + advice_text.replace('\n', '<br>')
            self.advice_label.setText(advice_html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TongueAnalyzerUI()
    window.show()
    sys.exit(app.exec_())