# -*- encoding: utf-8 -*-
'''
@File    :   paintView.py
@Time    :   2020/10/28 17:01:07
@Author  :   Haoyu Wang 
@Version :   1.1
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
from util.data import qpixmap2numpy, numpy2qpixmap

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
    isLocked = False
    isPressed = False
    isInitalized = False
    isPredicted = False
    # 显示相关参数
    isCursorChanged = True
    view_scale = 1.0
    alpha = [0x7f, 0x7f]
    # background = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    image = QImage(DEFAULT_SIZE, QImage.Format_ARGB32)
    mask_0 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # 生成的 prediction
    mask_1 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # 读取的 Ground Truth
    paint_0 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # negative paint
    paint_1 = QImage(DEFAULT_SIZE, QImage.Format_ARGB32) # positive paint
    seg_mask = None
    wheelSignal = pyqtSignal(int)
    wheelSignal2d = pyqtSignal(int)
    resizeSignal = pyqtSignal(int)
    saveSignal = pyqtSignal()
    
        
    def __init__(self, widgets):
        super().__init__(widgets)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        #! scrollBar On/off
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 设置初始的4层内容
        # self.background.fill(BACKGROUND_COLOR)
        # self.image.fill(IMAGE_DEFAULT_COLOR)
        self.mask_0.fill(MASK_DEFAULT_COLOR)
        self.mask_1.fill(MASK_DEFAULT_COLOR)
        self.layers = [self.image, self.mask_1, self.mask_0, self.paint_1, self.paint_0]
        # 其他属性
        self.offset = [0,0]
        self.colorTable = []
        self.paintIdx = 0
        
    def initalize(self, imageColor=IMAGE_DEFAULT_COLOR, name="unknown"):
        self.name = name
        self.image.fill(imageColor)
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(Qt.black)        
        self.colorTable = [qRgba(0,0,0,0x00), qRgba(0xff,0,0,int(self.alpha[1]))]
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
        # self.setCenter(4)
        self.setScene(self.scene)
        # self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.isInitalized = True
        self.autoRescale()

    def setItemOffset(self, offset):
        items = self.scene.items()
        for i in range(len(items)):
            items[i].setOffset(offset[0], offset[1])

    def addItemOffset(self, offset):
        items = self.scene.items()
        for i in range(len(items)):
            items[i].setOffset(self.offset[0]+offset[0], self.offset[1]+offset[1])

    def setCenter(self, item=4): 
        # return None
        # self.offset = [(W - H)/2, 0] if(W > H) else [0, (H - W)/2]
        
        # 添加长宽不一致图像的居中
        # bound = min(W, H)
        # self.offset = [(w - h)/2, 0] if(w > h) else [0, (h - w)/2]
        # if(bound>h): self.offset[1] += (bound-h)/2
        # elif(bound>w): self.offset[0] += (bound-w)/2

        W, H = self.viewport().width(), self.viewport().height()
        items = self.scene.items()
        w, h = items[4].pixmap().width()*self.view_scale, \
               items[4].pixmap().height()*self.view_scale
        self.offset[1] = (H-h)/(2*self.view_scale) if(H>h) else (h-H)/(2*self.view_scale)
        
        for i in range(len(items)):
            items[i].setOffset(self.offset[0], self.offset[1])

        

    def setOffset(self, x, y):
        center = (self.width()*0.5, self.height()*0.5)
        scenePos = self.mapToScene(center[0], center[1])
        viewPos = self.transform().map(scenePos)
        self.horizontalScrollBar().setValue(int(viewPos.x()-center[0]))
        self.verticalScrollBar().setValue(int(viewPos.y()-center[1]))

        # self.centerOn(x,y)
        # items = self.scene.items()
        # for i in range(len(items)):
        #     items[i].setOffset(x, y)

    def getOffset(self):
        # items = self.scene.items()
        # for i in range(len(items)):
        #     offset = items[i].offset()
        #     print(offset.x(), offset.y())
        return self.horizontalScrollBar().value(), self.verticalScrollBar().value()


    def setMode(self, m):
        self.mode = m
        self.isCursorChanged = True

    def setPenWidth(self,w):
        self.penWidth = w
        self.isCursorChanged = True

    def setAlpha(self, a, idx=1, r=0xff, g=0x00, b=0x00):
        if(not self.seg_mask): return 
        if(idx!=1 and idx!=0): return
        self.alpha[idx] = a
        # self.colorTable = [qRgba(0,0,0,0x00), qRgba(r,g,b,int(self.alpha[1]))]
        
        if(idx==1): self.mask_1 = self.getColoredMask(self.seg_mask)
        elif(idx==0): self.mask_0 = self.getColoredMask(self.seg_mask, r=89, g=99, b=210, a=a)
        mask = self.mask_1 if(idx==1) else self.mask_0

        self.scene.items()[2+idx].setPixmap(
            QPixmap.fromImage(mask).scaled(
                self.size(),
                aspectRatioMode = Qt.KeepAspectRatio,
                transformMode = Qt.SmoothTransformation
            ))
        # self.setCenter(2+idx)

    def setScale(self, s):   
        center = (self.viewport().width()*0.6, self.viewport().height()*0.6)
        scenePos = self.mapToScene(center[0], center[1])

        self.scale(s/self.view_scale, s/self.view_scale)
        self.view_scale = s
        
        viewPos = self.transform().map(scenePos)   
        self.horizontalScrollBar().setValue(int(viewPos.x()-center[0]))
        self.verticalScrollBar().setValue(int(viewPos.y()-center[1])) 
        return self.view_scale
        # self.setCenter(4)
        
        # self.horizontalScrollBar().setValue()
        # self.verticalScrollBar().setValue(int(self.verticalScrollBar().maximum()/2))        
        # self.horizontalScrollBar().setMaximum(self.height()*max(self.view_scale-1, 0))
        # self.verticalScrollBar().setMaximum(self.width()*max(self.view_scale-1, 0))

        # print(self.horizontalScrollBar().maximum(), ',', self.horizontalScrollBar().minimum(), ";",
        #       self.verticalScrollBar().maximum(), ",", self.verticalScrollBar().minimum(), ";")

    def autoRescale(self):
        self.view_scale = self.transform().m11()
        W, H = self.viewport().width(), \
               self.viewport().height()
        pix_w, pix_h = self.scene.items()[4].pixmap().width(), \
                       self.scene.items()[4].pixmap().height()
        new_view_scale = float(format(min(W/pix_w, H/pix_h), ".1f"))
        # trans = self.transform()
        # trans.scale(1.0, 1.0)
        # self.setTransform(trans)
        self.scale(new_view_scale/self.view_scale, new_view_scale/self.view_scale)
        self.view_scale = new_view_scale
        # self.horizontalScrollBar().setValue(0)
        # self.verticalScrollBar().setValue(0)
        return self.view_scale

    def setImage(self, img, keepRatio=True):
        mode = Qt.KeepAspectRatio if keepRatio else Qt.IgnoreAspectRatio
        self.image = img
        self.scene.items()[4].setPixmap(
            QPixmap.fromImage(self.image).scaled(
                self.size(), 
                aspectRatioMode = mode,
                transformMode = Qt.SmoothTransformation
            ))
        size = self.scene.items()[4].pixmap().size()
        # print(size)
        for i in range(4):
            self.layers[3-i].fill(QColor(0, 0, 0, 0))
            self.scene.items()[i].setPixmap(
                QPixmap.fromImage(self.layers[3-i]).scaled(
                    size, 
                    aspectRatioMode = Qt.IgnoreAspectRatio,
                    transformMode = Qt.SmoothTransformation
                ))

        # self.setCenter(4)
     
    def getColoredMask(self, seg_mask, r=0xff, g=0x00, b=0x00, a=0x7f):        
        seg_mask.setColorCount(2)     
        seg_mask.setColorTable([qRgba(0,0,0,0x00), qRgba(r,g,b,int(a))])
        seg_mask.convertToFormat(QImage.Format_ARGB32)
        return seg_mask

    def getPaint(self, idx=0, shape=None):
        pix = QPixmap(self.scene.items()[idx].pixmap()).copy()
        if(shape): pix = pix.scaled(shape[0], shape[1])
        return pix

    def setMask(self, seg, keepRatio=True, idx=1):
        mode = Qt.KeepAspectRatio if keepRatio else Qt.IgnoreAspectRatio
        self.seg_mask = seg.createMaskFromColor(0xFF000000, mode=Qt.MaskOutColor)
        if(idx==1): self.mask_1 = self.getColoredMask(self.seg_mask)
        elif(idx==0): self.mask_0 = self.getColoredMask(self.seg_mask, r=89, g=99, b=210)

        mask = self.mask_1 if(idx==1) else self.mask_0
        self.scene.items()[2+idx].setPixmap(
            QPixmap.fromImage(mask).scaled(
                self.scene.items()[4].pixmap().size(),
                aspectRatioMode = mode,
                transformMode = Qt.SmoothTransformation
            ))
        # self.setCenter(2+idx)

    def savePaint(self, idx=0):
        if(idx==2 and not self.isPredicted): return None, None
        else:
            items = self.scene.items()
            p_0, p_1 = items[idx+0].pixmap().copy(), items[idx+1].pixmap().copy()
            return p_0, p_1

    def loadPaint(self, p_0, p_1, idx=0):
        items = self.scene.items()
        # pix_0, pix_1 = numpy2qpixmap(p_0), numpy2qpixmap(p_1)
        if(idx==2):
            items[idx+0].setPixmap(p_0.scaled(items[idx+0].pixmap().size()))            
        else:
            pix_0, pix_1 = p_0, p_1
            items[idx+0].setPixmap(pix_0.scaled(items[idx+0].pixmap().size()))
            items[idx+1].setPixmap(pix_1.scaled(items[idx+1].pixmap().size()))

    def clearImage(self, imageColor=QColor(0,0,0,255)):
        self.image.fill(imageColor)
        self.setImage(self.image)

    def clearPaint(self, idx=0):
        size = self.scene.items()[4].pixmap().size()
        if(idx==0):
            self.scene.items()[idx+0].setPixmap(
                QPixmap.fromImage(self.paint_0).scaled(
                    size,
                    aspectRatioMode = Qt.IgnoreAspectRatio,
                    transformMode = Qt.SmoothTransformation
                ))
            self.scene.items()[idx+1].setPixmap(
                QPixmap.fromImage(self.paint_1).scaled(
                    size,
                    aspectRatioMode = Qt.IgnoreAspectRatio,
                    transformMode = Qt.SmoothTransformation
                ))
        elif(idx==2):
            self.mask_0.fill(QColor(0,0,0,0))
            self.scene.items()[idx+0].setPixmap(
                QPixmap.fromImage(self.mask_0).scaled(
                    size,
                    aspectRatioMode = Qt.IgnoreAspectRatio,
                    transformMode = Qt.SmoothTransformation
                ))
        # self.setCenter(4)


    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     self.autoRescale()

    def wheelEvent(self, event):
        zoomNum = 1 if event.angleDelta().y()<0 else -1
        self.wheelSignal.emit(zoomNum)
        self.wheelSignal2d.emit(zoomNum)


    def mousePressEvent(self, event):
        if(self.isLocked and self.mode != PaintMode.Drag): return None
        if(self.mode == PaintMode.Paint):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif(self.mode == PaintMode.Erase):
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(QCursor(QPixmap(":/橡皮光标.png")\
                                    .scaled(self.penWidth*self.view_scale,self.penWidth*self.view_scale, 
                                            transformMode = Qt.SmoothTransformation),-1,-1))
        elif(self.mode == PaintMode.Drag):
            # self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.OpenHandCursor)

        self.sPt = event.pos()
        self.ePt = event.pos()

        if(self.mode != PaintMode.Drag):
            if(event.button()==Qt.LeftButton):
                self.penColorIndex = 1
                self.paintIdx = 1
            elif(event.button()==Qt.RightButton):
                self.penColorIndex = 2
                self.paintIdx = 0
            else:
                self.penColorIndex = 0

            # line = [self.mapToScene(self.sPt)/self.view_scale, 
            #         self.mapToScene(self.ePt)/self.view_scale]
            line = [self.mapToScene(self.sPt), 
                    self.mapToScene(self.ePt)]
            img = QPixmap(self.scene.items()[self.paintIdx].pixmap())
            self.drawLine(line, img)
            # p_img.save("C:\\Users\\small\\Desktop\\img.png")
            self.scene.items()[self.paintIdx].setPixmap(img)
        self.isPressed = True

    def mouseReleaseEvent(self, event):
        self.isPressed = False
        self.saveSignal.emit()
        
    def mouseMoveEvent(self, event):
        # 修正图标形状
        if(self.isCursorChanged):            
            if(self.mode == PaintMode.Paint):
                self.setCursor(Qt.CrossCursor)
            elif(self.mode == PaintMode.Erase):
                self.setCursor(QCursor(QPixmap(":/橡皮光标.png")\
                                        .scaled(self.penWidth*self.view_scale,self.penWidth*self.view_scale, 
                                                transformMode = Qt.SmoothTransformation),-1,-1))
            elif(self.mode == PaintMode.Drag):
                self.setCursor(Qt.OpenHandCursor)
            self.isCursorChanged = False
        # ~ 鼠标移动定位记录
        # if(self.isLocked and self.mode != PaintMode.Drag): return None
        self.ePt = event.pos()

        # 绘画和擦除模式
        if(self.isPressed and self.mode!=PaintMode.Drag):
            # line = [self.mapToScene(self.sPt)/self.view_scale, 
            #         self.mapToScene(self.ePt)/self.view_scale]
            line = [self.mapToScene(self.sPt), 
                    self.mapToScene(self.ePt)]
            # if(event.button()==Qt.LeftButton):
            #     img = QPixmap(self.scene.items()[0].pixmap())
            #     self.drawLine(line, img)
            #     self.scene.items()[0].setPixmap(img)
            # elif(event.button()==Qt.RightButton):
            img = self.scene.items()[self.paintIdx].pixmap()
            self.drawLine(line, img)
            # img.save("C:\\Users\\small\\Desktop\\img{}.png".format(self.paintIdx))
            self.scene.items()[self.paintIdx].setPixmap(img)
        #拖拽模式
        if(self.isPressed and self.mode==PaintMode.Drag):
            scroll_val = self.ePt - self.sPt
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value()-scroll_val.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()-scroll_val.y())

        # ~ 鼠标移动定位更新
        self.sPt = self.ePt

    def drawLine(self, line, input_img):
        qp = QPainter(input_img)
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

    def resizeEvent(self, event):
        if(self.isInitalized): self.resizeSignal.emit(self.autoRescale())
        pass

    def screenSnap(self, filename):
        image = QImage(self.size(), QImage.Format_ARGB32);
        painter = QPainter(image)
        self.scene.render(painter)
        painter.end()
        size = self.scene.items()[4].pixmap().size()
        image = image.copy(self.offset[0], self.offset[1], 
                           size.width()*self.view_scale,
                           size.height()*self.view_scale).scaled(self.image.size())
        image.save(filename)