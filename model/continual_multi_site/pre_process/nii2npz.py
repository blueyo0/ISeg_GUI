#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/21 2:21 下午
# @Author  : Jingyang.Zhang
'''
处理nii数据，并将其转化为npz格式
'''
import shutil, os
import SimpleITK as sitk
import numpy as np

import platform
sysstr = platform.system()
if(sysstr=="Windows"):
    WORK_DIR = "D:/dataset/multi_site"
elif(sysstr=="Linux"):
    WORK_DIR = '/opt/data/private/why'
DATA_DIR = os.path.join(WORK_DIR, 'dataset')

def _load_normalized_vol(nii_path):
    '''
    从nii中读取体数据和分割标签
    :param nii_path: 单个病人的体数据和分割标签的路径
    :return: 体数据,分割和病人名称，注意是numpy数组格式
    '''
    img_path, label_path = nii_path.split(',')[0][3:], nii_path.split(',')[1][3:]
    img_path = os.path.join(WORK_DIR, img_path)
    label_path = os.path.join(WORK_DIR, label_path)
    case_name = img_path[-13:-7]
    print(f'Produce {case_name}')

    # 读入体数据和标注，进行正则化处理
    itk_img = sitk.ReadImage(img_path)
    itk_mask = sitk.ReadImage(label_path)
    img = sitk.GetArrayFromImage(itk_img)
    mask = sitk.GetArrayFromImage(itk_mask)
    binary_mask = np.ones(mask.shape)
    mean = np.sum(img * binary_mask) / np.sum(binary_mask)
    std = np.sqrt(np.sum(np.square(img - mean) * binary_mask) / np.sum(binary_mask))
    normalized_img = (img - mean) / std  # 正则化处理   
    mask[mask == 2] = 1

    # 旋转方向 和 数据验证
    normalized_img, mask = normalized_img.transpose([1,2,0]), mask.transpose([1,2,0])
    assert (normalized_img.shape[0] == 384) and (normalized_img.shape[1] == 384)
    assert (mask.shape[0] == 384) and (mask.shape[1] == 384)
    assert len(np.unique(mask)) == 2
    return normalized_img, mask, case_name

def _patch_center_z(mask):
    '''
    找到有分割结果的slice
    '''
    limX, limY, limZ = np.where(mask>0)
    z = np.arange(max(1, np.min(limZ)), min(np.max(limZ), mask.shape[2] - 2) + 1)
    return z


def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in range(1, num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)
    return one_hot


def _extract_patch(image, mask):
    '''
    :param image: 单个病人的体数据
    :param mask: 单个病人的分割标签
    :return: 输出数据增强之后的 patches 为了后续训练,
            image_patch-[384, 384, 3]; mask_patch-[384, 384, 2]
    '''
    slice_indexs = []
    image_patches = []
    mask_patches = []

    z = _patch_center_z(mask)
    for z_i in z:
        image_patch = image[:,:,z_i-1:z_i+2]
        mask_patch = mask[:,:,z_i]
        mask_patch = _label_decomp(mask_patch, 2)

        image_patches.append(image_patch)
        mask_patches.append(mask_patch)
        slice_indexs.append(z_i)
    return image_patches, mask_patches, slice_indexs


if __name__ == '__main__':
    datasets = ['BIDMC', 'HK', 'I2CVB' ,'ISBI', 'ISBI_1.5', 'UCL']
    for dataset in datasets:
        datalist = os.path.join(DATA_DIR, '%s_train_list' % dataset)
        # datalist = '../../dataset/%s_train_list' % dataset
        with open(datalist, 'r') as fp:
            rows = fp.readlines()
        image_list = [row[:-1] for row in rows]
        train_list = image_list[:int(len(image_list) * 0.75)]  # 训练数据nii名称列表
        test_list = image_list[int(len(image_list) * 0.75):]  # 测试数据nii名称列表

        output_train_dir = os.path.join(DATA_DIR, 'npz_data_new/%s/train' % dataset)  # 训练数据保存路径
        output_test_dir = os.path.join(DATA_DIR, 'npz_data_new/%s/test' % dataset) # 测试数据保存路径
        shutil.rmtree(output_train_dir, ignore_errors=True)
        os.makedirs(output_train_dir)
        shutil.rmtree(output_test_dir, ignore_errors=True)
        os.makedirs(output_test_dir)

        # 生成训练数据的 npz
        for nii_path in train_list:
            image, mask, case_name = _load_normalized_vol(nii_path)  # 单个病人的体数据和分割标注
            image_patches, mask_patches, slice_indexs = _extract_patch(image, mask)

            for i, slice_index in enumerate(slice_indexs):
                slice_name = '%s_Slice%d' % (case_name, slice_index)
                np.savez(os.path.join(output_train_dir, slice_name), image_patches[i], mask_patches[i])

        for nii_path in test_list:
            image, mask, case_name = _load_normalized_vol(nii_path)  # 单个病人的体数据和分割标注
            image_patches, mask_patches, slice_indexs = _extract_patch(image, mask)

            for i, slice_index in enumerate(slice_indexs):
                slice_name = '%s_Slice%d' % (case_name, slice_index)
                np.savez(os.path.join(output_test_dir, slice_name), image_patches[i], mask_patches[i])



