# -*- coding: utf-8 -*-

import sys
from PIL import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.mainWindow import *
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)

#-----------------------------------[ Slots ]----------------------------------#
	def openImageFile():
		image_path = QFileDialog.getOpenFileName(
			ui.centralwidget,
			"选择打开图像", 
			filter="Images(*.nii.gz)"
		)
		# 添加nib相关的读取
		image_3d = nib.load(image_path[0]).get_fdata()
		image_2d = image_3d[:,:,64]
		# plt.imshow(image_2d)
		# plt.show()
		image_2d = 255-image_2d
		# image_2d = image_2d.astype(np.uint16)
		for i in range(image_2d.shape[0]):
			for j in range(image_2d.shape[1]):
				if(image_2d[i,j]==255):
					# print(image_2d[i,j], i, j)
					image_2d[i,j]=0
		ui.view_a.setImage(Image.fromarray(np.uint8(image_2d)).rotate(90).transpose(Image.FLIP_TOP_BOTTOM).toqimage())
		# ui.view_a.setImage(QImage.fromData(image_2d, QImage.Format_Grayscale16))


#-----------------------------------[ Connect ]--------------------------------#
	ui.actionOpen_main_image.triggered.connect(openImageFile)


#-----------------------------------[ Novel end ]------------------------------#	
	myWin.show()
	ui.view_a.initalize()
	ui.view_s.initalize()
	ui.view_c.initalize()
	sys.exit(app.exec_())


