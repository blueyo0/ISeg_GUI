# -*- encoding: utf-8 -*-
'''
@File    :   simulate.py
@Time    :   2020/11/09 11:53:16
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   用户交互数据仿真生成
'''

import torch
import numpy as np
import random
from math import sqrt

SEED = 666
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')

from imageio import imwrite
from pathlib import Path
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QColor

def isTheSameColor(pixel_color: np.array, standard_color: QColor):
    '''判断bgra格式的ndarray color是否和对应Qcolor同色'''
    c1 = np.array(pixel_color)
    if(isinstance(standard_color, Qt.GlobalColor)): standard_color = QColor(standard_color)
    c2 = np.array([standard_color.blue(),
                   standard_color.green(),
                   standard_color.red(),
                   standard_color.alpha()])
    if(c1.shape!=c2.shape): return False
    return (c1==c2).all()

def isInMask(pt: QPoint, mask: np.array):
    if(isinstance(pt, QPoint)): ix, iy = pt.x(),pt.y()
    elif(isinstance(pt, list)): ix, iy = pt[0], pt[1]
    elif(isinstance(pt, tuple)): ix, iy = pt[0], pt[1]
    else: raise Exception("Invalid pt value!")
    if(ix>511 or ix<0 or iy>511 or iy<0): return False
    return isTheSameColor(mask[ix, iy], Qt.white)

def euclidDistance(p1: QPoint, p2: QPoint):
    '''两点间的欧氏距离的平方'''
    return (p1.x()-p2.x())**2 + (p1.y()-p2.y())**2

def randomSample(area: np.array, 
                 distance='euclid', pt_step=100.0, 
                 strategy='fixed', pt_num=10):
    '''
    @Brief: 根据输入的bgra数组，在白色区域内随机生成相互一定距离的一定数量的点
    :Param area: 可以生成点的区域，bgra格式
    :Param pt_step: 点与点之间的距离步长，计算方式使用distance指定的方式
    :Param pt_num: 如果strategy为fixed, 生成pt_num个点，为max则生成尽可能多的点
    @Return: 采样到的List(QPoint)
    '''
    pt_li = []
    for ix in range(area.shape[0]):
        for iy in range(area.shape[1]):
            if(isInMask(QPoint(ix, iy), area)):
            # if(isTheSameColor(area[ix,iy], Qt.white)):
                pt_li.append(QPoint(ix,iy))
    random.shuffle(pt_li)
    # 边界值处理和初始化
    if(len(pt_li)<1): return []
    if(len(pt_li)<2): return [pt_li[0]]
    if(pt_num<0): pt_num=10
    dist = euclidDistance if(distance=='euclid') else euclidDistance
    left_pt_num = pt_num if(strategy=='fixed') else len(pt_li)
    
    idx = 1
    sample_pts = [pt_li[0]]
    while(left_pt_num>0): # 找到足够的点跳出
        candidate = pt_li[idx]
        isSparseEnough = True    
        for pt in sample_pts:
            if(dist(pt, candidate)<pt_step):
                isSparseEnough = False
                break
        if(isSparseEnough): sample_pts.append(candidate)
        left_pt_num -= 1
        idx += 1
        if(idx>len(pt_li)-1): break # pt_li遍历完跳出

    return sample_pts

def isBoundaryReached(pt: QPoint, area: np.array, margin: int):
    '''从四个角进行简单探测，测试是否即将触碰边界'''
    pt_li = [
        QPoint(pt.x()+margin, pt.y()),
        QPoint(pt.x()-margin, pt.y()),
        QPoint(pt.x(), pt.y()-margin),
        QPoint(pt.x(), pt.y()+margin),
        QPoint(pt.x(), pt.y()),
    ]
    for p in pt_li:
        if(not isInMask(p, area)): return True
    return False

def getLinearPoints(start, end):
    pt_li = []
    if(end.x()==start.x()): return pt_li
    k = (end.y()-start.y()) / (end.x()-start.x())
    b = start.y()-k*start.x()
    x_range = np.arange(end.x()+1, start.x()) if(start.x() > end.x()) else np.arange(start.x()+1, end.x())
    for x in x_range:
        y = round(k*x+b)
        pt_li.append(QPoint(x,y))
    return pt_li

def randomScribble(pts: list, area: np.array,
                   margin=3,  noise=3,
                   scribble_num=3, connect_num=3):
    '''
    @Brief: 根据交互点列表，生成在area之外的点
    :Param pts: 交互点列表
    :Param area: 允许绘制的范围
    :Param margin: 绘制时遇到边界停止的距离
    :Param noise: 线条抖动范围
    :Param scribble_num: 生成的涂抹数量
    :Param connect_num: 每次连接的像素点数量
    @Return: 带有scribble的list(QPoint)
    '''
    if(len(pts)<3): return pts
    scribble_pts = []
    random.shuffle(pts)

    left_num, conn_num = scribble_num, connect_num
    idx_0, idx_1 = 0, 1
    while(left_num>0):
        if(idx_0 > len(pts)-1 or idx_1 > len(pts)-1): break
        start, end = pts[idx_0], pts[idx_1]
        line_pt = getLinearPoints(start, end)
        for pt in line_pt: 
            if(not isBoundaryReached(pt, area, margin)): scribble_pts.append(pt)
        conn_num -= 1
        if(conn_num<1): 
            conn_num = connect_num
            left_num-=1;    
            idx_0, idx_1 = idx_1+1, idx_1+2
        else:     
            idx_0, idx_1 = idx_1, idx_1+1


    return pts + scribble_pts

def getEuclidDistanceMap(pts: list, area: np.array, dim=4):
    result = np.zeros(area.shape, np.uint8)
    for ix in range(result.shape[0]):
        for iy in range(result.shape[1]):
            dist = [int(sqrt(euclidDistance(pt, QPoint(ix, iy)))) for pt in pts]
            min_dist = int(np.min(dist)) if(len(dist)>0) else 255
            if(min_dist>255): min_dist=255
            result[ix, iy] = np.array([min_dist,min_dist,min_dist,255]) if(dim==4) else 255
    return result




if __name__ == "__main__":
    start = QPoint(2,3)
    end = QPoint(10,10)
    li = getLinearPoints(start, end)
    print(li)
    pass