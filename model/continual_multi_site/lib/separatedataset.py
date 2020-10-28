#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 9:00 下午
# @Author  : Jingyang.Zhang
'''
dataset对应每个中心的数据
'''
from torch.utils.data import Dataset
import os, logging, torch
import numpy as np
import random
import imgaug.augmenters as iaa
#######
from torch.utils.data import DataLoader

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')
from pre_process.check_npz import check_npz
import matplotlib.pyplot as plt
#######

class SeparateDataset(Dataset):
    def __init__(self, npz_root_dir, mode, percent=1.0):
        self.npz_dir = os.path.join(npz_root_dir, mode)
        self.mode = mode
        self.slices = [file for file in os.listdir(self.npz_dir) if not file.startswith('.')]
        self.slices.sort()
        if percent < 1 and percent > 0:
            num = int(len(self.slices) * percent)
            start = random.randint(0, len(self.slices)-num)
            self.slices = self.slices[start:start+num]
        logging.info(f'Creating {self.mode}ing dataset in {self.npz_dir} with {len(self.slices)} slices in total.')
        # print(f'Creating dataset in {npz_dir} with {len(self.slices)} slices in total.')

    def __len__(self):
        return len(self.slices)

    def _translate(self, img, shift=10, roll=True):
        direction = ['right', 'left', 'down', 'up']
        i = random.randint(0, 3)
        img = img.copy()
        if direction[i] == 'right':
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:, :shift] = np.fliplr(right_slice)
        if direction[i] == 'left':
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        if direction[i] == 'down':
            down_slice = img[-shift:, :].copy()
            img[shift:, :] = img[:-shift, :]
            if roll:
                img[:shift, :] = down_slice
        if direction[i] == 'up':
            upper_slice = img[:shift, :].copy()
            img[:-shift, :] = img[shift:, :]
            if roll:
                img[-shift:, :] = upper_slice
        return img

    def _data_augmentation(self, img, gt):
        if random.randint(0,1) == 1:
            img = img[::-1,...]
            gt = gt[::-1,...] # 翻转img和gt中图片顺序
        # if random.randint(0,1) == 1:
        #     shift_pixel = random.randint(0,10)
        #     img = self._translate(img, shift=shift_pixel)
        #     gt = self._translate(gt, shift=shift_pixel)
        return img, gt


    def __getitem__(self, item):
        slice_name = os.path.join(self.npz_dir, self.slices[item])
        data = np.load(slice_name)
        img, gt = data['arr_0'], data['arr_1']
        if self.mode == 'train':
            img, gt = self._data_augmentation(img, gt)

        # 为pytorch转化维度和格式
        img, gt = img.transpose([2,0,1]), gt.transpose([2,0,1])
        img, gt = img.astype(np.float32), gt.astype(np.float32)
        return {'slice_name':self.slices[item], 'img':torch.from_numpy(img), 'gt':torch.from_numpy(gt)}


if __name__ == '__main__':

    dataset = SeparateDataset(npz_root_dir='../dataset/npz_split_data/I2CVB', mode='train', percent=1.0)
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4, pin_memory=True)
    temp = 1
    for batch in loader:
        print(temp)
        img, gt = batch['img'], batch['gt']
        plt.imsave('image-1.png', img[0,0,:, :], cmap='gray')
        plt.imsave('image.png', img[0,1,:, :], cmap='gray')
        plt.imsave('image+1.png', img[0,2,:, :], cmap='gray')
        plt.imsave('label.png', gt[0,1,:, :], cmap='gray')
        temp += 1

