from collections import namedtuple

import cv2
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider, QLabel

Size = namedtuple('Size', ['x', 'y'])
SIZE = Size(10, 10)
ENERGY_HL = 30
ENERGY_LL = 10

energy_field = np.zeros((SIZE.x, SIZE.y)) + \
    np.expand_dims(np.flip(np.linspace(start=ENERGY_LL, stop=ENERGY_HL, num=SIZE.y, dtype=np.uint8)), axis=1)


class Gene:
    def __init__(self):
        self.code = np.zeros((1, 4))


class Genome:
    def __init__(self):
        pass


class Tree:
    def __init__(self):
        pass



class Environment:
    def __init__(self):
        pass





class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.image_view = QLabel()

        self.COLORTABLE = []
        for i in range(256):
            self.COLORTABLE.append(QtGui.qRgb(i / 4, i, i / 2))
        self.spectroWidth = 512
        self.spectroHeight = 256
        self.data = self.create_data()

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.image_view)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer()
        self.timer.start(1000)
        self.timer.timeout.connect(self.update_data)
        # self.central_widget.connect(self.timer, QtCore.SIGNAL('timeout()'), self.update_data)

    def create_data(self):
        a = np.random.random(self.spectroHeight * self.spectroWidth) * 255
        a = np.reshape(a, (self.spectroHeight, self.spectroWidth))
        # a = np.require(a, np.uint8, 'C')
        return a.astype(np.uint8)

    def update_image(self):
        pass

    def update_data(self):
        self.data = np.roll(self.data, -5)
        QI = QtGui.QImage(self.data.data, self.spectroWidth, self.spectroHeight, QtGui.QImage.Format_Indexed8)
        QI.setColorTable(self.COLORTABLE)
        self.image_view.setPixmap(QtGui.QPixmap.fromImage(QI))


class MovieThread(QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def run(self):
        self.camera.acquire_movie(200)


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())


