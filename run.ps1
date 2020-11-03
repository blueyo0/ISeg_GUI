# 开始编译
conda activate iseg
pyuic5 -o ui/mainWindow.py ui/mainWindow.ui
pyrcc5 -o rsc_rc.py Resources/rsc.qrc
python main.py

# 清空上次编译结果
# rm ./ui/mainWindow.py
# rm rsc_rc.py 