from data import *
from simulate import *
from PIL import Image, ImageQt
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QImage, QPixmap, qRgb
import cv2
import numpy as np
import time

def testQImage():
    # 此示例图来自https://doc.qt.io/qt-5.14/qimage.html
    image = QImage(3,3, QImage.Format_ARGB32)
    image.fill(QColor(0,0,0,0))
    value = qRgb(189, 149, 39); # 0xffbd9527
    image.setPixel(1, 1, value);
    value = qRgb(122, 163, 39); # 0xff7aa327
    image.setPixel(0, 1, value);
    image.setPixel(1, 0, value);
    value = qRgb(237, 187, 51); # 0xffedba31
    image.setPixel(2, 1, value);
    return image

def imageShow(img):
    # show QImage as PIL
    pil_img = ImageQt.fromqimage(img)
    pil_img = pil_img.resize((300, 300), Image.BOX)
    pil_img.show()

def cv_show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   #cv2.destroyWindow(wname)

def testPosiGen():
    img_path = "D:\\dataset\\KITS2019\\data\\case_00000\\segmentation.nii.gz"
    img_3d = load_nii_data(img_path)
    img = arr2img(img_3d[:,:,255]).createMaskFromColor(0xFF000000, mode=Qt.MaskOutColor)\
                                  .convertToFormat(QImage.Format_ARGB32)
    arr = qimage2numpy(img)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    arr_2 = cv2.erode(arr, kernel, iterations=5)
    arr_3 = cv2.multiply(arr_2, 0.7)
    arr = cv2.subtract(arr, arr_3)
    # cv_show(arr)

    start = time.time() 
    li = randomSample(arr_2)
    print("random sample %d points with time cost:%.1f s" % (len(li), time.time()-start))
    for pt in li:
        arr[pt.x(), pt.y()] = np.array([0,255,0,255])
        arr[pt.x()+1, pt.y()-1] = np.array([0,255,0,255])
        arr[pt.x()+1, pt.y()+1] = np.array([0,255,0,255])
        arr[pt.x()-1, pt.y()+1] = np.array([0,255,0,255])
        arr[pt.x()-1, pt.y()-1] = np.array([0,255,0,255])
    cv_show(arr)

def testCV():
    # img = cv2.imread("D:/code/ISeg_igst/ISeg_igst/util/1.PNG", cv2.IMREAD_UNCHANGED)
    # img = testQImage()
    # img = QImage("D:/code/ISeg_igst/ISeg_igst/util/1.PNG")
    img = QImage("D:/code/ISeg_igst/ISeg_igst/util/seg_00007.png")
    img = img.createMaskFromColor(0xFF000000, mode=Qt.MaskOutColor)\
             .convertToFormat(QImage.Format_ARGB32)
    img = qimage2numpy(img)

    # kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img_2 = cv2.dilate(img, kernel, iterations=10)
    # img = cv2.erode(img, kernel, iterations=10)
    # img = cv2.subtract(img_2, img)

    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    cv_show(img)


if __name__ == '__main__':
    testPosiGen()
    pass
