# -*- encoding: utf-8 -*-
'''
@File    :   paintView.py
@Time    :   2020/10/28 17:01:07
@Author  :   Haoyu Wang 
@Version :   1.0
@Contact :   small_dark@sina.com
'''
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.paintMode import PaintMode

import time

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
    alpha = 0xff
    background = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    image = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    mask = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    paint_0 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # negative paint
    paint_1 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # positive paint
    wheelSignal = pyqtSignal(int)
        
    def __init__(self, widgets):
        super().__init__(widgets)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 设置初始的4层内容
        # self.background.fill(BACKGROUND_COLOR)
        self.image.fill(IMAGE_DEFAULT_COLOR)
        self.mask.fill(MASK_DEFAULT_COLOR)
        self.layers = [self.image, self.mask, self.paint_1, self.paint_0]
        # 其他属性
        self.offset = [0,0]
        self.colorTable = []

    def initalize(self):
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(Qt.black)        
        self.colorTable = [qRgba(0,0,0,0x00), qRgba(0xff,0,0,int(self.alpha))]
        # self.scene.addItem(QGraphicsPixmapItem(
        #     QPixmap.fromImage(self.background.scaled(self.size()*1.5))
        # ))

        # 获取最大边和最小边的值
        for img in self.layers:
            self.scene.addItem(QGraphicsPixmapItem(
                QPixmap.fromImage(img).scaled(
                    self.size(),
                    aspectRatioMode = Qt.KeepAspectRatio
                )
            ))
        self.setCenter()
        self.setScene(self.scene)
        self.autoRescale()

    def setCenter(self, item=3):        
        W, H = self.size().width(), self.size().height()
        self.offset = [(W - H)/2, 0] if(W > H) else [0, (H - W)/2]
        items = self.scene.items()
        # 添加长宽不一致图像的居中
        w, h = items[item].pixmap().width(), \
               items[item].pixmap().height()
        if(w>h): self.offset[1] += (w-h)/2
        else: self.offset[0] += (h-w)/2

        w, h = w*self.scale, h*self.scale
        if(w>W and h>H): #item无法完整显示时，针对当前中心放缩
            pass
            # self.offset[0] += (W-w)/2
            # self.offset[1] += (H-h)/2
        elif(w<W and h<H): # item较小时，居中显示
            self.offset[0] = (W-w)/2 / self.scale
            self.offset[1] = (H-h)/2 / self.scale

        for i in range(len(items)):
            items[i].setOffset(self.offset[0], self.offset[1])
        # print(self.offset[0], self.offset[1])
        # print(self.mapToScene(QPoint(W/2, H/2))/self.scale)
 

    def setMode(self, m):
        self.mode = m
        self.isCursorChanged = True

    def setPenWidth(self,w):
        self.penWidth = w
        self.isCursorChanged = True

    def setAlpha(self, a):
        self.alpha =a
        self.colorTable = [qRgba(0,0,0,0x00), qRgba(0xff,0,0,int(self.alpha))]
        self.mask = self.getColoredMask(self.seg_mask)

        self.scene.items()[2].setPixmap(
            QPixmap.fromImage(self.mask).scaled(
                self.size(),
                aspectRatioMode = Qt.KeepAspectRatio,
                transformMode = Qt.SmoothTransformation
            ))
        self.setCenter(2)

    def setScale(self, s):        # 为所有item设置scale
        self.scale = s
        for item in self.scene.items():
            item.setScale(s)
        if(self.mode == PaintMode.Erase):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(QCursor(QPixmap(":/橡皮光标.png")\
                                    .scaled(self.penWidth*self.scale,
                                            self.penWidth*self.scale, 
                                            transformMode = Qt.SmoothTransformation),-1,-1))
        self.setCenter()
        self.horizontalScrollBar().setMaximum(self.height()*max(self.scale-1, 0))
        self.verticalScrollBar().setMaximum(self.width()*max(self.scale-1, 0))

        # print(self.horizontalScrollBar().maximum(), ',', self.horizontalScrollBar().minimum(), ";",
        #       self.verticalScrollBar().maximum(), ",", self.verticalScrollBar().minimum(), ";")

    def autoRescale(self):
        W, H = self.width(), self.height()
        pix_w, pix_h = self.scene.items()[0].pixmap().width(), \
                       self.scene.items()[0].pixmap().height()
        self.scale = float(format(min(W/pix_w, H/pix_h), ".1f"))
        self.setScale(self.scale)
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        return self.scale

    def setImage(self, img, keepRatio=True):
        mode = Qt.KeepAspectRatio if keepRatio else Qt.IgnoreAspectRatio
        self.image = img
        self.scene.items()[3].setPixmap(
            QPixmap.fromImage(self.image).scaled(
                self.size(), 
                aspectRatioMode = mode,
                transformMode = Qt.SmoothTransformation
            ))
        self.setCenter(3)
     
    def getColoredMask(self, seg_mask):        
        seg_mask.setColorCount(2)     
        seg_mask.setColorTable(self.colorTable)
        seg_mask.convertToFormat(QImage.Format_ARGB32)
        return seg_mask


    def setMask(self, seg, keepRatio=True):
        mode = Qt.KeepAspectRatio if keepRatio else Qt.IgnoreAspectRatio
        self.seg_mask = seg.createMaskFromColor(0xFF000000, mode=Qt.MaskOutColor)
        self.mask = self.getColoredMask(self.seg_mask)

        self.scene.items()[2].setPixmap(
            QPixmap.fromImage(self.mask).scaled(
                self.size(),
                aspectRatioMode = mode,
                transformMode = Qt.SmoothTransformation
            ))
        self.setCenter(2)

    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     self.autoRescale()

    def wheelEvent(self, event):
        zoomNum = 1 if event.angleDelta().y()<0 else -1
        self.wheelSignal.emit(zoomNum)

    def mousePressEvent(self, event):
        if(self.mode == PaintMode.Paint):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif(self.mode == PaintMode.Erase):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(QCursor(QPixmap(":/橡皮光标.png")\
                                    .scaled(self.penWidth*self.scale,self.penWidth*self.scale, 
                                            transformMode = Qt.SmoothTransformation),-1,-1))
        elif(self.mode == PaintMode.Drag):
            # self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.OpenHandCursor)

        self.sPt = event.pos()
        self.ePt = event.pos()

        paintIdx = 0
        if(self.mode != PaintMode.Drag):
            if(event.button()==Qt.LeftButton):
                self.penColorIndex = 1
                paintIdx = 1
            elif(event.button()==Qt.RightButton):
                self.penColorIndex = 2
            else:
                self.penColorIndex = 0

            line = [self.mapToScene(self.sPt)/self.scale, 
                    self.mapToScene(self.ePt)/self.scale]
            img = QPixmap(self.scene.items()[paintIdx].pixmap())
            self.drawLine(line, img)
            self.scene.items()[paintIdx].setPixmap(img)
        self.isPressed = True

    def mouseReleaseEvent(self, event):
        self.isPressed = False
        
    def mouseMoveEvent(self, event):
        # 修正图标形状
        if(self.isCursorChanged):            
            if(self.mode == PaintMode.Paint):
                self.setCursor(Qt.CrossCursor)
            elif(self.mode == PaintMode.Erase):
                self.setCursor(QCursor(QPixmap(":/橡皮光标.png")\
                                        .scaled(self.penWidth*self.scale,self.penWidth*self.scale, 
                                                transformMode = Qt.SmoothTransformation),-1,-1))
            elif(self.mode == PaintMode.Drag):
                self.setCursor(Qt.OpenHandCursor)
            self.isCursorChanged = False
        # ~ 鼠标移动定位记录
        self.ePt = event.pos()

        # 绘画和擦除模式
        if(self.isPressed and self.mode!=PaintMode.Drag):
            line = [self.mapToScene(self.sPt)/self.scale, 
                    self.mapToScene(self.ePt)/self.scale]
            img = QPixmap(self.scene.items()[0].pixmap())
            self.drawLine(line, img)
            self.scene.items()[0].setPixmap(img)
            img = QPixmap(self.scene.items()[1].pixmap())
            self.drawLine(line, img)
            self.scene.items()[1].setPixmap(img)
        #拖拽模式
        if(self.isPressed and self.mode==PaintMode.Drag):
            scroll_val = self.ePt - self.sPt
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value()-scroll_val.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()-scroll_val.y())

        # ~ 鼠标移动定位更新
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