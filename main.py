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
	# myWin.showFullScreen()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)
	myWin.setWindowState(Qt.WindowMaximized)

#------------------------------[ Inter Variables ]-----------------------------#
	ui.general_mode = PaintMode.Paint
	ui.view_dim = 3
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
	ui.pred_3d = [{},{}]
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
			filter="Image(*.nii.gz *.png *.tif *.jpg)"
		)
		if(len(image_path[0])<1):
			return None
		path_li = image_path[0].split(".")
		if(path_li[-1]=='gz'):	
			ui.view_a.initalize(imageColor=QColor(0, 0, 0, 0), name="a")
			ui.view_s.initalize(imageColor=QColor(0, 0, 0, 0), name="s")
			ui.view_c.initalize(imageColor=QColor(0, 0, 0, 0), name="c")
			# ui.view_a.resetTransform(); ui.view_s.resetTransform(); ui.view_c.resetTransform();
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
			ui.view_a.setCenter(); ui.view_s.setCenter(); ui.view_c.setCenter();
			ui.view_3d.clearImage(); ui.view_3d.clearPaint()
			if(ui.view_dim==2): changeViewTo3d()
		else:
			if(ui.view_dim==3): changeViewTo2d()
			ui.view_3d.initalize(imageColor=QColor(0, 0, 0, 255))
			ui.view_3d.setImage(QImage(image_path[0]))
			ui.view_3d.setCenter()
			ui.view_a.clearImage(); ui.view_a.clearPaint()
			ui.view_s.clearImage(); ui.view_s.clearPaint()
			ui.view_c.clearImage(); ui.view_c.clearPaint()

		autoRescale()
		ui.action_open_seg.setEnabled(True)
		ui.action_predict.setEnabled(True)
		ui.action_refine.setEnabled(True)


	def openMaskFile():
		mask_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开分割结果", 
			filter="Segmentation(*.nii.gz *jpg *png *gif *tif)"
		)
		if(len(mask_path[0])<1):
			return None
		path_li = mask_path[0].split(".")
		if(path_li[-1]=='gz'):
			if(ui.view_dim!=3): QMessageBox(QMessageBox.Warning, "错误", "请打开正确的分割结果").exec()
			if(ui.view_dim==2): changeViewTo3d()
			ui.seg_3d, ui.direction_seg = load_nii_data(mask_path[0], getDirection=True)
			updateSegSlice()
		else:
			if(ui.view_dim!=2): QMessageBox(QMessageBox.Warning, "错误", "请打开正确的分割结果").exec()
			if(ui.view_dim==3): changeViewTo2d()
			ui.view_3d.setMask(QImage(mask_path[0]))


		autoRescale()
		# ui.view_a.setMask(Image.fromarray(np.uint8(image_2d)).rotate(90).transpose(Image.FLIP_TOP_BOTTOM).toqimage())

	def updateXPosByWheel(val):
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.pos_x.setValue(ui.pos_x.value()+val)


	def updateYPosByWheel(val):
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.pos_y.setValue(ui.pos_y.value()+val)


	def updateZPosByWheel(val):
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
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
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.verticalScrollBar_s.setValue(ui.pos_y.value())

	def updateVerticalScrollBarByZ():
		if(not ui.isPaintSaved): savePaint(ui.verticalScrollBar_a.value())
		ui.verticalScrollBar_c.setValue(ui.pos_z.value())

	def updateToolBox(g_mode):
		for btn, mode in ui.toolDict.items():
			if(mode == g_mode): btn.setChecked(True)
			else: btn.setChecked(False)

	def updatePaintMode():	
		ui.view_a.setMode(ui.general_mode)
		ui.view_s.setMode(ui.general_mode)
		ui.view_c.setMode(ui.general_mode)
		if(ui.view_dim==2): ui.view_3d.setMode(ui.general_mode)

	def updateScale2d(val):
		# if(ui.view_dim!=2): return None
		if(ui.scaleSpinBox.value()>5): val *= 2
		scale = int(ui.scaleSpinBox.value()*10+val)*0.1
		if(scale>0.0 and scale<10.0):
			s = ui.view_3d.setScale(scale)
			ui.scaleSpinBox.setValue(s)

	def updateScale():
		if(ui.view_dim==2): ui.view_3d.setScale(ui.scaleSpinBox.value())
		if(ui.view_dim==3):
			ui.view_a.setScale(ui.scaleSpinBox.value())
			ui.view_s.setScale(ui.scaleSpinBox.value())
			ui.view_c.setScale(ui.scaleSpinBox.value())
	
	def autoRescale():
		s1 = ui.view_a.autoRescale()
		s2 = ui.view_s.autoRescale()
		s3 = ui.view_c.autoRescale()
		if(ui.view_dim==2): s1 = ui.view_3d.autoRescale()	
		# ui.scaleSpinBox.disconnect(updateScale)
		ui.scaleSpinBox.setValue(s1)
		# ui.scaleSpinBox.valueChanged.connect(updateScale)

	def resizeScale(val):
		ui.scaleSpinBox.valueChanged.disconnect(updateScale)
		ui.scaleSpinBox.setValue(val)
		ui.scaleSpinBox.valueChanged.connect(updateScale)

	def updateASlider():
		ui.alphaHSlider.setValue(ui.alphaSpinBox.value())

	def updateASpinBox():
		ui.alphaSpinBox.setValue(ui.alphaHSlider.value())
		ui.view_a.setAlpha(int(2.55*ui.alphaSpinBox.value()))
		ui.view_s.setAlpha(int(2.55*ui.alphaSpinBox.value()))
		ui.view_c.setAlpha(int(2.55*ui.alphaSpinBox.value()))
		if(ui.view_dim==2): ui.view_3d.setAlpha(int(2.55*ui.alphaSpinBox.value()))

	def updateASlider_2():
		ui.alphaHSlider_2.setValue(ui.alphaSpinBox_2.value())

	def updateASpinBox_2():
		ui.alphaSpinBox_2.setValue(ui.alphaHSlider_2.value())
		ui.view_a.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)
		ui.view_s.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)
		ui.view_c.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)
		if(ui.view_dim==2): ui.view_3d.setAlpha(int(2.55*ui.alphaSpinBox_2.value()), idx=0)

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
	
	def updateOffset():
		ui.view_c.setOffset(ui.offset_x.value(), ui.offset_y.value())

	def synOffset():
		x, y = ui.view_c.getOffset()
		ui.offset_x.setValue(x), ui.offset_y.setValue(y)

	def updatePenWidth():
		ui.penWidthList[ui.penWidthIdx] = ui.penWidthSpinBox.value()
		ui.view_a.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_s.setPenWidth(ui.penWidthSpinBox.value())
		ui.view_c.setPenWidth(ui.penWidthSpinBox.value())
		if(ui.view_dim==2): ui.view_3d.setPenWidth(ui.penWidthSpinBox.value())

	def savePaint(val):
		if(len(ui.paint_3d)<1): return None
		val = val-1
		p0, p1 = ui.view_a.savePaint()
		if(val in ui.paint_3d[0]): ui.paint_3d[0].update({val:p0})
		else: ui.paint_3d[0][val] = p0
		if(val in ui.paint_3d[1]): ui.paint_3d[1].update({val:p1})
		else: ui.paint_3d[1][val] = p1

		p0, p1 = ui.view_a.savePaint(2)
		if(p0 and p1):
			if(val in ui.pred_3d[0]): ui.pred_3d[0].update({val:p0})
			else: ui.pred_3d[0][val] = p0
			ui.view_a.isPredicted = False
		ui.isPaintSaved = True

	def loadPaint(val):
		if(len(ui.paint_3d)<2 and len(ui.pred_3d)<2): return None
		val = val-1
		if(val in ui.paint_3d[0] and val in ui.paint_3d[1]):
			p0 = ui.paint_3d[0][val]
			p1 = ui.paint_3d[1][val]
			ui.view_a.loadPaint(p0, p1)
		else:
			ui.view_a.clearPaint()
		
		if(val in ui.pred_3d[0]):
			p0 = ui.pred_3d[0][val]
			ui.view_a.loadPaint(p0, p1, 2)
		else:
			ui.view_a.clearPaint(2)
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
		ui.view_a.isPredicted = True
		ui.isPaintSaved = False
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
		ui.view_a.isPredicted = True
		ui.isPaintSaved = False
		pass

	def testMessageBox():
		msgBox = QMessageBox(QMessageBox.Information, "测试", "test function")
		msgBox.exec()
		pass

	def changeViewTo2d():
		ui.view_3d.wheelSignal2d.connect(updateScale2d)
		ui.view_dim=2
		ui.view_3d.isLocked = False
		ui.label.hide(); ui.pos_x.hide(); ui.pos_y.hide(); ui.pos_z.hide();
		ui.view_a.hide(); ui.toolButton_a.hide(); ui.verticalScrollBar_a.hide()
		ui.view_s.hide(); ui.toolButton_s.hide(); ui.verticalScrollBar_s.hide()
		ui.view_c.hide(); ui.toolButton_c.hide(); ui.verticalScrollBar_c.hide()
		ui.gridLayout.setColumnStretch(0,1)
		ui.gridLayout.setRowStretch(0,1)
		ui.gridLayout.setColumnStretch(1,0)
		ui.gridLayout.setRowStretch(1,0)
	
	def changeViewTo3d():
		ui.view_3d.wheelSignal2d.disconnect(updateScale2d)
		ui.view_dim=3
		ui.view_3d.isLocked = True
		ui.view_s.isLocked = True
		ui.view_c.isLocked = True
		ui.label.show(); ui.pos_x.show(); ui.pos_y.show(); ui.pos_z.show();
		ui.view_a.show(); ui.toolButton_a.show(); ui.verticalScrollBar_a.show()
		ui.view_s.show(); ui.toolButton_s.show(); ui.verticalScrollBar_s.show()
		ui.view_c.show(); ui.toolButton_c.show(); ui.verticalScrollBar_c.show()
		ui.gridLayout.setColumnStretch(0,1)
		ui.gridLayout.setColumnStretch(1,1)
		ui.gridLayout.setRowStretch(0,1)
		ui.gridLayout.setRowStretch(1,1)

	def changeViewDimension():
		if(ui.view_dim==3): changeViewTo2d()
		elif(ui.view_dim==2): changeViewTo3d()

	def clearOnePaint():
		ui.view_a.clearPaint()
		ui.view_3d.clearPaint()

	def saveUpdate():
		ui.isPaintSaved = False

#-----------------------------------[ Connect ]--------------------------------#
	ui.action_open_main_image.triggered.connect(openImageFile)
	ui.action_open_seg.triggered.connect(openMaskFile)
	ui.action_predict.triggered.connect(predict)
	ui.action_refine.triggered.connect(refine)
	ui.action_clear_paint.triggered.connect(clearOnePaint)
	# ui.action_snap_dist_map.triggered.connect(testMessageBox)

	ui.view_a.wheelSignal.connect(updateXPosByWheel)
	ui.view_s.wheelSignal.connect(updateYPosByWheel)
	ui.view_c.wheelSignal.connect(updateZPosByWheel)
	ui.view_3d.wheelSignal2d.connect(updateScale2d)
	ui.view_a.resizeSignal.connect(resizeScale)
	ui.view_s.resizeSignal.connect(resizeScale)
	ui.view_c.resizeSignal.connect(resizeScale)
	ui.view_a.saveSignal.connect(saveUpdate)

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

	#! testButton
	ui.testButton.hide()
	# ui.testButton.clicked.connect(changeViewDimension)
	# ui.action_snap_dist_map.triggered.connect(changeViewTo3d)

	#! offset debug button
	ui.offset_synButton.hide(); ui.offset_updButton.hide(); ui.offset_x.hide(); ui.offset_y.hide()
	# ui.offset_synButton.clicked.connect(synOffset)
	# ui.offset_updButton.clicked.connect(updateOffset)

	ui.toolButton_a.clicked.connect(testMessageBox)
	ui.toolButton_s.clicked.connect(testMessageBox)
	ui.toolButton_c.clicked.connect(testMessageBox)

#-----------------------------------[ Novel end ]------------------------------#	
	myWin.show()
	ui.view_a.initalize(name="a")
	ui.view_s.initalize(name="s")
	ui.view_c.initalize(name="c")
	ui.view_3d.initalize(imageColor=QColor(0, 0, 0, 255))
	ui.view_3d.isLocked = True
	ui.view_s.isLocked = True
	ui.view_c.isLocked = True
	# ui.scaleSpinBox.setFocusPolicy(Qt.NoFocus)
	# 坐标显示和scrollbar同步
	updatePos()
	sys.exit(app.exec_())


