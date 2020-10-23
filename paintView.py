# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from paintMode import PaintMode

DEFAULT_SIZE = QSize(256,256)
BACKGROUND_COLOR = QColor(0,0,0,255)
IMAGE_DEFAULT_COLOR = QColor(100, 100, 100, 255)
MASK_DEFAULT_COLOR = QColor(200, 200, 200, 0)


class PaintView(QtWidgets.QGraphicsView):
    # 绘制相关参数
    penWidth = 5
    penColorIndex = 0
    penColor = [QColor(0, 0, 0, 0), # 空白色
                QColor(0, 255, 0, 255), 
                QColor(255, 255, 0, 255)] # positive绿色， negative黄色
    sPt = QPoint()
    ePt = QPoint()
    mode = PaintMode.Paint
    isPressed = False
    # 显示相关参数
    isCursorChanged = True
    scale = 1.0
    background = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    image = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    mask = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    paint = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
        
    def __init__(self, widgets):
        super().__init__(widgets)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 设置初始的4层内容
        self.background.fill(BACKGROUND_COLOR)
        self.image.fill(IMAGE_DEFAULT_COLOR)
        self.mask.fill(MASK_DEFAULT_COLOR)
        self.layers = [self.image, self.mask, self.paint]
    
    def initalize(self):
        self.scene = QGraphicsScene(self)
        self.scene.addItem(QGraphicsPixmapItem(
            QPixmap.fromImage(self.background.scaled(self.size()))
        ))

        # 获取最大边和最小边的值
        size = self.size()
        self.offset = ((size.width() - size.height())/2, 0) \
                 if(size.width() > size.height()) \
                 else (0, (size.height() - size.width())/2)

        for img in self.layers:
            self.scene.addItem(QGraphicsPixmapItem(
                QPixmap.fromImage(img).scaled(
                    self.size(), 
                    aspectRatioMode = Qt.KeepAspectRatio
                )
            ))
        items = self.scene.items()
        for i in range(len(items)-1):
            items[i].setOffset(self.offset[0], self.offset[1])
        self.setScene(self.scene)

    def setMode(self, m):
        self.mode = m
        self.isCursorChanged = True

    def setPenWidth(self,w):
        self.penWidth = w
        self.isCursorChanged = True

    def setScale(self, s):
        self.scale = s
        for item in self.scene.items():
            item.setScale(s)
        if(self.mode == PaintMode.Erase):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(QCursor(QPixmap(":/Resources/橡皮光标.png")\
                                    .scaled(self.penWidth*self.scale,self.penWidth*self.scale, 
                                            transformMode = Qt.SmoothTransformation),-1,-1))

    def setImage(self, img):
        self.image = img
        self.scene.items()[2].setPixmap(
            QPixmap.fromImage(self.image).scaled(
                self.size(), 
                aspectRatioMode = Qt.KeepAspectRatio,
                transformMode = Qt.SmoothTransformation
            ))

    def setMask(self, seg):
        self.mask = seg.convertToFormat(QImage.Format_ARGB32)
        f = self.mask.format()
        for ix in range(self.mask.width()):
            for iy in range(self.mask.height()):
                if(ix==-65536 or iy==-65536):
                    print("error")
                if(self.mask.pixel(ix, iy)%0x1000000==0x000000):
                    self.mask.setPixel(ix, iy, 0x00000000)
                else:
                    self.mask.setPixel(ix, iy, 0xFFFF0000)

                # if(self.mask.pixel(ix, iy)==QRGB()):
        self.scene.items()[1].setPixmap(
            QPixmap.fromImage(self.mask).scaled(
                self.size(), 
                aspectRatioMode = Qt.KeepAspectRatio,
                transformMode = Qt.SmoothTransformation
            ))


    def mousePressEvent(self, event):
        if(self.mode == PaintMode.Paint):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif(self.mode == PaintMode.Erase):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(QCursor(QPixmap(":/Resources/橡皮光标.png")\
                                    .scaled(self.penWidth*self.scale,self.penWidth*self.scale, 
                                            transformMode = Qt.SmoothTransformation),-1,-1))

        if(event.button()==Qt.LeftButton):
            self.penColorIndex = 1
        elif(event.button()==Qt.RightButton):
            self.penColorIndex = 2
        else:
            self.penColorIndex = 0

        self.sPt = event.pos()
        self.ePt = event.pos()
        line = [self.mapToScene(self.sPt)/self.scale, 
                self.mapToScene(self.ePt)/self.scale]
        img = QPixmap(self.scene.items()[0].pixmap())
        self.drawLine(line, img)
        self.scene.items()[0].setPixmap(img)
        self.isPressed = True

    def mouseReleaseEvent(self, event):
        # self.setCursor(Qt.ArrowCursor)
        self.isPressed = False
        
    def mouseMoveEvent(self, event):
        # 修正图标形状
        if(self.isCursorChanged):            
            if(self.mode == PaintMode.Paint):
                self.setCursor(Qt.CrossCursor)
            elif(self.mode == PaintMode.Erase):
                self.setCursor(QCursor(QPixmap(":/Resources/橡皮光标.png")\
                                        .scaled(self.penWidth*self.scale,self.penWidth*self.scale, 
                                                transformMode = Qt.SmoothTransformation),-1,-1))
            self.isCursorChanged = False
        # 鼠标按下，开始执行功能
        if(self.isPressed):
            self.ePt = event.pos()
            line = [self.mapToScene(self.sPt)/self.scale, 
                    self.mapToScene(self.ePt)/self.scale]
            img = QPixmap(self.scene.items()[0].pixmap())
            self.drawLine(line, img)
            self.scene.items()[0].setPixmap(img)
            self.sPt = self.ePt

    def drawLine(self, line, img):
        qp = QPainter(img)
        if(self.mode==PaintMode.Paint):
            qp.setCompositionMode(QPainter.CompositionMode_SourceOver)
        elif(self.mode==PaintMode.Erase):
            qp.setCompositionMode(QPainter.CompositionMode_Clear)
        
        # QPen：适合画轮廓线
        pen = QPen()
        pen.setColor(self.penColor[self.penColorIndex])       
        pen.setWidth(self.penWidth)
        qp.setPen(pen)

        qp.drawLine(line[0]-QPointF(self.offset[0], self.offset[1]), 
                    line[1]-QPointF(self.offset[0], self.offset[1]))
        qp.end()
        self.scene.items()[0].setPixmap(img)
        self.sPt = self.ePt