import argparse
import sys

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


class ImageLabel(QLabel):
    def __init__(self, parent, filename):
        super().__init__(parent)
        self.original_pixmap = QPixmap(filename)  # Replace with your image path
        self.set_image()

        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

    def set_image(self):
        window_size = self.parent().size()
        scaled_pixmap = self.original_pixmap.scaled(window_size, Qt.KeepAspectRatio)
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.set_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            self.update()
            print("Top left coordinates: {}".format(self.start_point))
            print("Bottom right coordinates: {}".format(self.end_point))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        painter.drawRect(
            self.start_point.x(),
            self.start_point.y(),
            self.end_point.x() - self.start_point.x(),
            self.end_point.y() - self.start_point.y(),
        )


class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.setWindowTitle("Box Drawing")
        self.setFixedSize(900, 600)  # Set fixed window size

        self.image_label = ImageLabel(self, filename)
        self.setCentralWidget(self.image_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="path to the image to use")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.filename)
    window.show()
    sys.exit(app.exec_())
