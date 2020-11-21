# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/10/28 17:00:29
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   主函数
'''
import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.mainWindow import *

from ui.paintMode import PaintMode
from util.data import load_nii_data
from util.data import arr2img

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)

#------------------------------[ Inter Variables ]-----------------------------#
	ui.general_mode = PaintMode.Paint
	ui.penWidthList = [5, 15, 0]
	ui.penWidthIdx = 0
	ui.toolDict = { 
		ui.paintButton : PaintMode.Paint,
		ui.eraseButton : PaintMode.Erase,
		ui.dragButton  : PaintMode.Drag
	}
	ui.img_3d = np.array([])
	ui.direction_img = None
	ui.seg_3d = np.array([])
	ui.direction_seg = None
	ui.paint_3d = [{},{}]
	ui.isSizeLoad = False
	ui.isPaintSaved = False
	ui.currPos = [0,0,0]

	ui.counter = [0, 0]
#-----------------------------------[ Slots ]----------------------------------#
	def updateImgSlice():
		if(ui.img_3d.ndim<3): return None
		# ui.view_a.setImage(ui.img_3d[0][ui.verticalScrollBar_a.value()-1]) 
		# ui.view_s.setImage(ui.img_3d[1][ui.verticalScrollBar_s.value()-1]) 
		# ui.view_c.setImage(ui.img_3d[2][ui.verticalScrollBar_c.value()-1]) 
	
		ui.view_a.setImage(arr2img(ui.img_3d[:,:,ui.verticalScrollBar_a.value()-1], position='a', direction=ui.direction_img)) 
		ui.view_s.setImage(arr2img(ui.img_3d[ui.verticalScrollBar_s.value()-1,:,:], position='s', direction=ui.direction_img)) 
		ui.view_c.setImage(arr2img(ui.img_3d[:,ui.verticalScrollBar_c.value()-1,:], position='c', direction=ui.direction_img)) 
	
	def updateSegSlice():
		if(ui.seg_3d.ndim<3): return None
		ui.view_a.setMask(arr2img(ui.seg_3d[:,:,ui.verticalScrollBar_a.value()-1], position='a', direction=ui.direction_seg)) 
		ui.view_s.setMask(arr2img(ui.seg_3d[ui.verticalScrollBar_s.value()-1,:,:], position='s', direction=ui.direction_seg)) 
		ui.view_c.setMask(arr2img(ui.seg_3d[:,ui.verticalScrollBar_c.value()-1,:], position='c', direction=ui.direction_seg)) 


	def openImageFile():
		image_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开图像", 
			filter="Image(*.nii.gz)"
		)
		if(len(image_path[0])<1):
			return None

		ui.img_3d, ui.direction_img = load_nii_data(image_path[0], getDirection=True)
		# if(not ui.isSizeLoad):
		ui.img_size = [ui.img_3d.shape[2], ui.img_3d.shape[0], ui.img_3d.shape[1]]	
		ui.paint_3d = [{},{}]
		ui.verticalScrollBar_a.setMaximum(ui.img_size[0])
		ui.verticalScrollBar_s.setMaximum(ui.img_size[1])
		ui.verticalScrollBar_c.setMaximum(ui.img_size[2])
		ui.pos_x.setMaximum(ui.img_size[0])
		ui.pos_y.setMaximum(ui.img_size[1])
		ui.pos_z.setMaximum(ui.img_size[2])
			
		ui.verticalScrollBar_a.setValue(int(ui.img_size[0]/2))
		ui.verticalScrollBar_s.setValue(int(ui.img_size[1]/2))
		ui.verticalScrollBar_c.setValue(int(ui.img_size[2]/2))
		ui.isSizeLoad = True
		updateImgSlice()
		autoRescale()
		ui.actionopen_seg.setEnabled(True)


	def openMaskFile():
		mask_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开分割结果", 
			filter="Segmentation(*.nii.gz)"
		)
		if(len(mask_path[0])<1):
			return None

		# ui.seg_3d = load_nii_data(mask_path[0])
		ui.seg_3d, ui.direction_seg = load_nii_data(mask_path[0], getDirection=True)
		if(not ui.isSizeLoad):
			ui.img_size = [ui.seg_3d.shape[2], ui.seg_3d.shape[0], ui.seg_3d.shape[1]]	

			ui.verticalScrollBar_a.setMaximum(ui.img_size[0])
			ui.verticalScrollBar_s.setMaximum(ui.img_size[1])
			ui.verticalScrollBar_c.setMaximum(ui.img_size[2])
			ui.pos_x.setMaximum(ui.img_size[0])
			ui.pos_y.setMaximum(ui.img_size[1])
			ui.pos_z.setMaximum(ui.img_size[2])
			
			ui.verticalScrollBar_a.setValue(int(ui.img_size[0]/2))
			ui.verticalScrollBar_s.setValue(int(ui.img_size[1]/2))
			ui.verticalScrollBar_c.setValue(int(ui.img_size[2]/2))
		else:		
			updateSegSlice()
		autoRescale()
		# ui.view_a.setMask(Image.fromarray(np.uint8(image_2d)).rotate(90).transpose(Image.FLIP_TOP_BOTTOM).toqimage())

	def updateXPosByWheel(val):
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.pos_x.setValue(ui.pos_x.value()+val)


	def updateYPosByWheel(val):
		ui.pos_y.setValue(ui.pos_y.value()+val)


	def updateZPosByWheel(val):
		ui.pos_z.setValue(ui.pos_z.value()+val)


	def updatePos():
		ui.pos_x.setValue(ui.verticalScrollBar_a.value())
		ui.pos_y.setValue(ui.verticalScrollBar_s.value())
		ui.pos_z.setValue(ui.verticalScrollBar_c.value())
		ui.currPos = [ui.pos_x.value(), ui.pos_y.value(), ui.pos_z.value()]
		updateImgSlice()
		updateSegSlice()

	def updatePosByX():
		if(not ui.isPaintSaved): savePaint(ui.pos_x.value())
		ui.pos_x.setValue(ui.verticalScrollBar_a.value())
		ui.currPos = [ui.pos_x.value(), ui.pos_y.value(), ui.pos_z.value()]
		updateImgSlice()
		updateSegSlice()
		loadPaint(ui.currPos[0])

	def updatePosByY():
		ui.pos_y.setValue(ui.verticalScrollBar_s.value())
		ui.currPos = [ui.pos_x.value(), ui.pos_y.value(), ui.pos_z.value()]
		updateImgSlice()
		updateSegSlice()

	def updatePosByZ():
		ui.pos_z.setValue(ui.verticalScrollBar_c.value())
		ui.currPos = [ui.pos_x.value(), ui.pos_y.value(), ui.pos_z.value()]
		updateImgSlice()
		updateSegSlice()

	def updateVerticalScrollBar():
		ui.verticalScrollBar_a.setValue(ui.pos_x.value())
		ui.verticalScrollBar_s.setValue(ui.pos_y.value())
		ui.verticalScrollBar_c.setValue(ui.pos_z.value())

	def updateVerticalScrollBarByX():
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.verticalScrollBar_a.setValue(ui.pos_x.value())

	def updateVerticalScrollBarByY():
		ui.verticalScrollBar_s.setValue(ui.pos_y.value())

	def updateVerticalScrollBarByZ():
		ui.verticalScrollBar_c.setValue(ui.pos_z.value())

	def updateToolBox(g_mode):
		for btn, mode in ui.toolDict.items():
			if(mode == g_mode):
				btn.setChecked(True)
			else:
				btn.setChecked(False)

	def updatePaintMode():	
		ui.view_a.setMode(ui.general_mode)
		ui.view_s.setMode(ui.general_mode)
		ui.view_c.setMode(ui.general_mode)


	def updateScale():
		ui.view_a.setScale(ui.scaleSpinBox.value())
		ui.view_s.setScale(ui.scaleSpinBox.value())
		ui.view_c.setScale(ui.scaleSpinBox.value())
	
	def autoRescale():
		s1 = ui.view_a.autoRescale()
		s2 = ui.view_s.autoRescale()
		s3 = ui.view_c.autoRescale()	
		ui.scaleSpinBox.setValue(s1)

	def updateASlider():
		ui.alphaHSlider.setValue(ui.alphaSpinBox.value())

	def updateASpinBox():
		ui.alphaSpinBox.setValue(ui.alphaHSlider.value())
		ui.view_a.setAlpha(int(2.55*ui.alphaSpinBox.value()))
		ui.view_s.setAlpha(int(2.55*ui.alphaSpinBox.value()))
		ui.view_c.setAlpha(int(2.55*ui.alphaSpinBox.value()))


	def changeModeToPaint():
		ui.penWidthIdx = 0
		ui.general_mode = PaintMode.Paint
		ui.state_label.setText("标记")
		ui.label_3.setVisible(True)
		ui.penWidthSpinBox.setVisible(True)
		ui.penWidthSpinBox.setValue(ui.penWidthList[ui.penWidthIdx])
		updateToolBox(PaintMode.Paint)
		updatePaintMode()

	def changeModeToErase():
		ui.penWidthIdx = 1
		ui.general_mode = PaintMode.Erase
		ui.state_label.setText("擦除")
		ui.label_3.setVisible(True)
		ui.penWidthSpinBox.setVisible(True)
		ui.penWidthSpinBox.setValue(ui.penWidthList[ui.penWidthIdx])
		updateToolBox(PaintMode.Erase)
		updatePaintMode()

	def changeModeToDrag():
		# ui.penWidthIdx = 2
		ui.general_mode = PaintMode.Drag
		ui.state_label.setText("拖拽")
		ui.label_3.setVisible(False)
		ui.penWidthSpinBox.setVisible(False)
		# ui.penWidthSpinBox.setValue(ui.penWidthList[ui.penWidthIdx])
		updateToolBox(PaintMode.Drag)
		updatePaintMode()
	
	def updatePenWidth():
		ui.penWidthList[ui.penWidthIdx] = ui.penWidthSpinBox.value()
		ui.view_a.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_s.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_c.setPenWidth(ui.penWidthSpinBox.value())

	def savePaint(val):
		if(len(ui.paint_3d)<2): return None
		# ~ for test
		# val = ui.currPos[0]

		val = val-1
		p0, p1 = ui.view_a.savePaint()
		if(val in ui.paint_3d[0]): ui.paint_3d[0].update({val:p0})
		else: ui.paint_3d[0][val] = p0
		if(val in ui.paint_3d[1]): ui.paint_3d[1].update({val:p1})
		else: ui.paint_3d[1][val] = p1
		ui.isPaintSaved = True

	def loadPaint(val):
		if(len(ui.paint_3d)<2): return None
		val = val-1
		if(val in ui.paint_3d[0] and val in ui.paint_3d[1]):
			p0 = ui.paint_3d[0][val]
			p1 = ui.paint_3d[1][val]
			ui.view_a.loadPaint(p0, p1)
		else:
			ui.view_a.clearPaint()
		ui.isPaintSaved = False
		

	def clearPaint():
		ui.view_a.clearPaint()


#-----------------------------------[ Connect ]--------------------------------#
	ui.actionOpen_main_image.triggered.connect(openImageFile)
	ui.actionopen_seg.triggered.connect(openMaskFile)

	ui.view_a.wheelSignal.connect(updateXPosByWheel)
	ui.view_s.wheelSignal.connect(updateYPosByWheel)
	ui.view_c.wheelSignal.connect(updateZPosByWheel)
	ui.verticalScrollBar_a.valueChanged.connect(updatePosByX)
	ui.verticalScrollBar_s.valueChanged.connect(updatePosByY)
	ui.verticalScrollBar_c.valueChanged.connect(updatePosByZ)
	ui.pos_x.valueChanged.connect(updateVerticalScrollBarByX)
	ui.pos_y.valueChanged.connect(updateVerticalScrollBarByY)
	ui.pos_z.valueChanged.connect(updateVerticalScrollBarByZ)
	ui.alphaHSlider.valueChanged.connect(updateASpinBox)
	ui.alphaSpinBox.valueChanged.connect(updateASlider)
	ui.penWidthSpinBox.valueChanged.connect(updatePenWidth)
	ui.scaleSpinBox.valueChanged.connect(updateScale)
	ui.autoScaleButton.clicked.connect(autoRescale)
	# ui.autoScaleButton.clicked.connect(clearPaint)
	# ui.autoScaleButton.clicked.connect(savePaint)

	ui.paintButton.clicked.connect(changeModeToPaint)
	ui.eraseButton.clicked.connect(changeModeToErase)
	ui.dragButton.clicked.connect(changeModeToDrag)
	# ui.dragButton.clicked.connect(loadPaint)

#-----------------------------------[ Novel end ]------------------------------#	
	myWin.show()
	ui.view_a.initalize()
	ui.view_s.initalize()
	ui.view_c.initalize()
	# ui.scaleSpinBox.setFocusPolicy(Qt.NoFocus)
	# 坐标显示和scrollbar同步
	updatePos()
	sys.exit(app.exec_())


