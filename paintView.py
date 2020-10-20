# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class PaintView(QtWidgets.QGraphicsView):

    def __init__(self,dialog):
        super().__init__(dialog)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.show()
    
        



