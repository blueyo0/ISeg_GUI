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
import time

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.mainWindow import *

from ui.paintMode import PaintMode
from util.data import load_nii_data
from util.data import arr2img, qpixmap2numpy
import model.model_util as mutil
from util.simulate import getEuclidDistanceMap
from util.test import cv_show, cv_show_with_sim
import cv2

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)

#------------------------------[ Inter Variables ]-----------------------------#
	ui.general_mode = PaintMode.Paint
	ui.penWidthList = [1, 15, 0]
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
	# ！ model preload
	ui.pnet_model = mutil.load_net_model(type='pnet')
	ui.rnet_model = mutil.load_net_model(type='rnet')
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
		ui.verticalScrollBar_a.setMaximum(ui.img_size[0]-1)
		ui.verticalScrollBar_a.setMinimum(1)
		ui.verticalScrollBar_s.setMaximum(ui.img_size[1]-1)
		ui.verticalScrollBar_s.setMinimum(1)
		ui.verticalScrollBar_c.setMaximum(ui.img_size[2]-1)
		ui.verticalScrollBar_c.setMinimum(1)
		ui.pos_x.setMaximum(ui.img_size[0])
		ui.pos_y.setMaximum(ui.img_size[1])
		ui.pos_z.setMaximum(ui.img_size[2])
			
		ui.verticalScrollBar_a.setValue(int(ui.img_size[0]/2))
		ui.verticalScrollBar_s.setValue(int(ui.img_size[1]/2))
		ui.verticalScrollBar_c.setValue(int(ui.img_size[2]/2))
		ui.isSizeLoad = True
		updateImgSlice()
		autoRescale()
		ui.action_open_seg.setEnabled(True)
		ui.action_predict.setEnabled(True)
		ui.action_refine.setEnabled(True)


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
			if(mode == g_mode): btn.setChecked(True)
			else: btn.setChecked(False)

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

	def updateASlider_2():
		ui.alphaHSlider_2.setValue(ui.alphaSpinBox_2.value())

	def updateASpinBox_2():
		ui.alphaSpinBox_2.setValue(ui.alphaHSlider_2.value())
		ui.view_a.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)
		ui.view_s.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)
		ui.view_c.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)

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

	def predict():
		img_patchs = ui.img_3d[:,:,ui.verticalScrollBar_a.value()-2:ui.verticalScrollBar_a.value()+1].copy()
		for i in range(3):
			img_single = img_patchs[:,:,i]
			mean = np.mean(img_single)
			std = np.std(img_single)
			normalized_img = (img_single - mean) / std  # 正则化处理  
			img_patchs[:,:,i] = normalized_img
		pred = mutil.predict(ui.pnet_model, img_patchs)
		ui.view_a.setMask(arr2img(pred, position='a'), idx=0) 
		# TO-DO: mask_0的三维化

	def refine():
		paint_c = [qpixmap2numpy(ui.view_a.getPaint().scaled(512,512)), qpixmap2numpy(ui.view_a.getPaint(1).scaled(512,512))]
		paint = [cv2.cvtColor(paint_c[0], cv2.COLOR_BGRA2GRAY), cv2.cvtColor(paint_c[1], cv2.COLOR_BGRA2GRAY)]
		# cv_show(paint[0])
		# cv_show(paint[1])
		img = ui.img_3d[:,:,ui.verticalScrollBar_a.value()-2:ui.verticalScrollBar_a.value()+1].copy()
		for i in range(3):
			img_single = img[:,:,i]
			mean = np.mean(img_single)
			std = np.std(img_single)
			normalized_img = (img_single - mean) / std  # 正则化处理  
			img[:,:,i] = normalized_img
		start = time.time()
		li, shape = [[],[]], paint[0].shape
		for pidx in range(2):
			val = np.max(paint[pidx])
			for ix in range(shape[0]):
				for iy in range(shape[1]):
					if(paint[pidx][iy,ix]==val): li[pidx].append(QPoint(ix,iy))
		# cv_show(paint[0])
		# cv_show(paint[1])
		# cv_show_with_sim(img[:,:,1].copy(), li[0], li[1], dim=1)

		print("time cost:%.1f s" % (time.time()-start))
		map_0 = getEuclidDistanceMap(li[0], img[:,:,0], dim=1)
		print("map_0 with time cost:%.1f s" % (time.time()-start))
		map_1 = getEuclidDistanceMap(li[1], img[:,:,0], dim=1)
		print("map_1 with time cost:%.1f s" % (time.time()-start))
		map_0 = map_0.astype(np.float)/255
		map_1 = map_1.astype(np.float)/255

		np.savez("D:/code/ISeg_igst/ISeg_GUI/model/input_save/map_0.npz", map_0)
		np.savez("D:/code/ISeg_igst/ISeg_GUI/model/input_save/map_1.npz", map_1)
		# cv_show(map_1)
		# cv_show(map_0)

		sim = np.array([map_0,map_1]).transpose([1,2,0])
		img_patchs = np.concatenate((img,sim),axis=2)

		pred = mutil.predict(ui.rnet_model, img_patchs)
		ui.view_a.setMask(arr2img(pred, position='a'), idx=0) 
		# TO-DO: 使用四叉树处理点的距离，绘制欧几里得距离图 
		pass

	def testMessageBox():
		msgBox = QMessageBox(QMessageBox.Information, "测试", "test function")
		msgBox.exec()
		pass

#-----------------------------------[ Connect ]--------------------------------#
	ui.action_open_main_image.triggered.connect(openImageFile)
	ui.action_open_seg.triggered.connect(openMaskFile)
	ui.action_predict.triggered.connect(predict)
	ui.action_refine.triggered.connect(refine)
	ui.action_snap_dist_map.triggered.connect(testMessageBox)

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
	ui.alphaHSlider_2.valueChanged.connect(updateASpinBox_2)
	ui.alphaSpinBox_2.valueChanged.connect(updateASlider_2)

	ui.penWidthSpinBox.valueChanged.connect(updatePenWidth)
	ui.scaleSpinBox.valueChanged.connect(updateScale)
	ui.autoScaleButton.clicked.connect(autoRescale)

	ui.paintButton.clicked.connect(changeModeToPaint)
	ui.eraseButton.clicked.connect(changeModeToErase)
	ui.dragButton.clicked.connect(changeModeToDrag)

	ui.toolButton_a.clicked.connect(testMessageBox)
	ui.toolButton_s.clicked.connect(testMessageBox)
	ui.toolButton_c.clicked.connect(testMessageBox)

#-----------------------------------[ Novel end ]------------------------------#	
	myWin.show()
	ui.view_a.initalize()
	ui.view_s.initalize()
	ui.view_c.initalize()
	# ui.scaleSpinBox.setFocusPolicy(Qt.NoFocus)
	# 坐标显示和scrollbar同步
	updatePos()
	sys.exit(app.exec_())


