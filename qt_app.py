import argparse
import sys

import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


class ImageLabel(QLabel):
    def __init__(self, parent, pixmap):
        super().__init__(parent)
        self.original_pixmap = pixmap
        self.scaled_pixmap = pixmap
        self.setPixmap(self.scaled_pixmap)

        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

    def resizeEvent(self, event):
        # Maintain aspect ratio of image when resizing
        self.scaled_pixmap = self.original_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(self.scaled_pixmap)

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

            # Calculate the scale factors for width and height
            scale_width = self.original_pixmap.width() / self.scaled_pixmap.width()
            scale_height = self.original_pixmap.height() / self.scaled_pixmap.height()

            # Apply the scale factors to the start and end points
            scaled_start_x = self.start_point.x() * scale_width
            scaled_start_y = self.start_point.y() * scale_height
            scaled_end_x = self.end_point.x() * scale_width
            scaled_end_y = self.end_point.y() * scale_height

            # Order the coordinates to ensure xmin/xmax and ymin/ymax are correct
            xmin = int(round(min(scaled_start_x, scaled_end_x)))
            ymin = int(round(min(scaled_start_y, scaled_end_y)))
            xmax = int(round(max(scaled_start_x, scaled_end_x)))
            ymax = int(round(max(scaled_start_y, scaled_end_y)))

            # Print out the coordinates
            print(
                "Coordinates: xmin = {}, ymin = {}, xmax = {}, ymax = {}".format(
                    xmin, ymin, xmax, ymax
                )
            )
            print("bbox: {} {} {} {}".format(xmin, ymin, xmax, ymax))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing:
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

        # Load the image
        original_pixmap = QPixmap(filename)

        # Calculate a reasonable window size, not exceeding the screen size
        screen_size = QApplication.primaryScreen().size()
        window_size = original_pixmap.size().scaled(screen_size, Qt.KeepAspectRatio)

        # Set the window size and initialize the ImageLabel
        self.setFixedSize(window_size)
        self.image_label = ImageLabel(self, original_pixmap)
        self.setCentralWidget(self.image_label)

        # Center the window on the screen
        self.centerWindow()

    def centerWindow(self):
        # Center the MainWindow on the screen
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, required=True, help="Path to the image file."
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args.filename)
    window.show()
    sys.exit(app.exec_())
