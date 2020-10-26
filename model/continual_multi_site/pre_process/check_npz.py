#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/21 3:07 下午
# @Author  : Jingyang.Zhang
'''
显示npz预处理结果
'''
import numpy as np
import matplotlib.pyplot as plt

def check_npz(image, label):
    '''
    显示 image 和 label 的结果，保存成结果
    '''
    plt.imsave('image-1.png', image[:, :, 0], cmap='gray')
    plt.imsave('image.png', image[:, :, 1], cmap='gray')
    plt.imsave('image+1.png', image[:, :, 2], cmap='gray')
    plt.imsave('label.png', label[:, :, 1], cmap='gray')

if __name__ == '__main__':
    data = np.load('../../dataset/npz_data/BIDMC/Case00_Slice2.npz')
    image = data['arr_0']
    label = data['arr_1']
    check_npz(image, label)