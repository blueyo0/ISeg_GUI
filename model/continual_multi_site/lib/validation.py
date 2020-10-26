#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 2:49 下午
# @Author  : Jingyang.Zhang
'''
计算slice-level和case-level的dice coefficient
'''
import torch
from torch.utils.data import DataLoader
from lib.losses import dice_loss
import numpy as np

def validation(model, loader_val:DataLoader):
    '''
    :param model: 分割模型
    :param loader_val: 测试数据的Dataloder
    :return: slice-level dice, case-level dice
    '''
    val_dice_slice = {}
    val_dice_case = {}

    # 计算 slice_level 的 dice coefficient
    for batch in loader_val:
        slice_name, img, gt = batch['slice_name'], batch['img'].cuda(), batch['gt'].cuda()
        with torch.no_grad():
            model.eval()
            pred = model(img)
        val_dice_slice[slice_name[0]] = 1 - dice_loss((pred[:, 1, :, :] >= 0.5).float(), gt[:, 1, :, :].float()).item()

    # 计算 case_level 的 dice_coefficient
    case_names = np.unique([v.split('_')[0] for v in val_dice_slice.keys()])
    for case_name in case_names:
        val_dice_case[case_name] = []  # 每个case初始化为空列表
    for slice_name, slice_dice in val_dice_slice.items():
        val_dice_case[slice_name.split('_')[0]].append(slice_dice)  # 每个case的slice dice添加入列表

    return np.mean([np.mean(v) for v in val_dice_case.values()])
    # return np.mean([v for v in val_dice_slice.values()]), np.mean([np.mean(v) for v in val_dice_case.values()])