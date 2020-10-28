# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/10/28 17:00:29
@Author  :   Haoyu Wang 
@Version :   1.0
@Contact :   small_dark@sina.com
'''

import sys
from PIL import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.mainWindow import *
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from paintMode import PaintMode

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)

#------------------------------[ Inter Variables ]-----------------------------#
	ui.general_mode = PaintMode.Paint
	ui.penWidthList = [5, 15]
	ui.penWidthIdx = 0
	ui.toolDict = { 
		ui.paintButton : PaintMode.Paint,
		ui.eraseButton : PaintMode.Erase
	}

#-----------------------------------[ Slots ]----------------------------------#
	def openImageFile():
		image_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开图像", 
			filter="Images(*.nii.gz)"
		)
		if(len(image_path[0])<1):
			return
		# 添加nib相关的读取
		image_3d = nib.load(image_path[0]).get_fdata()
		ui.verticalScrollBar_a.setMaximum(image_3d.shape[2])
		ui.verticalScrollBar_s.setMaximum(image_3d.shape[0])
		ui.verticalScrollBar_c.setMaximum(image_3d.shape[1])
		image_2d = image_3d[:,:,ui.verticalScrollBar_a.value()]
		image_2d = 255-image_2d
		# image_2d = image_2d.astype(np.uint16)
		for i in range(image_2d.shape[0]):
			for j in range(image_2d.shape[1]):
				if(image_2d[i,j]>=253):
					# print(image_2d[i,j], i, j)
					image_2d[i,j]=0
		ui.view_a.setImage(Image.fromarray(np.uint8(image_2d)).rotate(90).transpose(Image.FLIP_TOP_BOTTOM).toqimage())

	def openMaskFile():
		mask_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开分割结果", 
			filter="Segmentation(*.nii.gz)"
		)
		if(len(mask_path[0])<1):
			return
		# 添加nib相关的读取
		image_3d = nib.load(mask_path[0]).get_fdata()
		image_2d = image_3d[:,:,ui.verticalScrollBar_a.value()]
		image_2d = 255-image_2d
		# image_2d = image_2d.astype(np.uint16)
		for i in range(image_2d.shape[0]):
			for j in range(image_2d.shape[1]):
				if(image_2d[i,j]>=253):
					# print(image_2d[i,j], i, j)
					image_2d[i,j]=0
		ui.view_a.setMask(Image.fromarray(np.uint8(image_2d)).rotate(90).transpose(Image.FLIP_TOP_BOTTOM).toqimage())

	def updatePos():
		ui.pos_x.setValue(ui.verticalScrollBar_a.value())
		ui.pos_y.setValue(ui.verticalScrollBar_s.value())
		ui.pos_z.setValue(ui.verticalScrollBar_c.value())

	def updateVerticalScrollBar():
		ui.verticalScrollBar_a.setValue(ui.pos_x.value())
		ui.verticalScrollBar_s.setValue(ui.pos_y.value())
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

	def changeModeToPaint():
		ui.penWidthIdx = 0
		ui.general_mode = PaintMode.Paint
		ui.state_label.setText("标记")
		ui.penWidthSpinBox.setValue(ui.penWidthList[ui.penWidthIdx])
		updateToolBox(PaintMode.Paint)
		updatePaintMode()

	def changeModeToErase():
		ui.penWidthIdx = 1
		ui.general_mode = PaintMode.Erase
		ui.state_label.setText("擦除")
		ui.penWidthSpinBox.setValue(ui.penWidthList[ui.penWidthIdx])
		updateToolBox(PaintMode.Erase)
		updatePaintMode()

	def updatePenWidth():
		ui.penWidthList[ui.penWidthIdx] = ui.penWidthSpinBox.value()
		ui.view_a.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_s.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_c.setPenWidth(ui.penWidthSpinBox.value())

#-----------------------------------[ Connect ]--------------------------------#
	ui.actionOpen_main_image.triggered.connect(openImageFile)
	ui.actionopen_seg.triggered.connect(openMaskFile)

	ui.verticalScrollBar_a.valueChanged.connect(updatePos)
	ui.verticalScrollBar_s.valueChanged.connect(updatePos)
	ui.verticalScrollBar_c.valueChanged.connect(updatePos)
	ui.pos_x.valueChanged.connect(updateVerticalScrollBar)
	ui.pos_y.valueChanged.connect(updateVerticalScrollBar)
	ui.pos_z.valueChanged.connect(updateVerticalScrollBar)
	ui.penWidthSpinBox.valueChanged.connect(updatePenWidth)
	ui.scaleSpinBox.valueChanged.connect(updateScale)

	ui.paintButton.clicked.connect(changeModeToPaint)
	ui.eraseButton.clicked.connect(changeModeToErase)

#-----------------------------------[ Novel end ]------------------------------#	
	myWin.show()
	ui.view_a.initalize()
	ui.view_s.initalize()
	ui.view_c.initalize()
	ui.scaleSpinBox.setFocusPolicy(Qt.NoFocus)
	# 坐标显示和scrollbar同步
	updatePos()
	sys.exit(app.exec_())


