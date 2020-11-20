# -*- encoding: utf-8 -*-
'''
@File    :   simulate.py
@Time    :   2020/11/09 11:53:16
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   function of simulating user-interaction
'''

import torch
import numpy as np
import random

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
            if(isTheSameColor(area[ix,iy], Qt.white)):
                pt_li.append(QPoint(ix,iy))
    random.shuffle(pt_li)
    # 边界值处理和初始化
    if(len(pt_li)<1): return []
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


def generateInteraction(
        img: np.array, gt: np.array, 
        distance = euclidDistance, 
        step = 10, margin = 10):
    '''
    @Brief: 根据输入的图像以及ground truth分割进行
    TO-DO: 交互过程的整体函数，最后再写
    '''
    pass

def generateP1(
        gt: np.array, dis=euclidDistance, 
        step=10, margin=10):
    '''
    @Brief: 生成positive interaction(P1)
    :param gt: ground truth arr(数据格式为bgra)
    @Return: 
    '''    
    
    pass






if __name__ == "__main__":
    pass