# -*- coding: utf-8 -*-

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from mainWindow import *


if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(myWin)
	myWin.show()
	sys.exit(app.exec_())