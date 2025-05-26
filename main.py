import sys
import os
from PyQt5.QtWidgets import QApplication

from config import create_directories
from src.ui.tongue_analyzer_ui import TongueAnalyzerUI


def main():
    # 确保所有必要的目录存在
    create_directories()

    # 启动ui
    app = QApplication(sys.argv)
    window = TongueAnalyzerUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()