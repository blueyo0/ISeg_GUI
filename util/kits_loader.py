# -*- encoding: utf-8 -*-
'''
@File    :   kits_loader.py
@Time    :   2020/10/30 09:47:01
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
'''

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision import transforms
import tables
from pathlib import Path
import sys
import os
from data import KITS_SLICE_h5, KITS_SLICE_sim
from test import cv_show

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





def get_loader(data_dir, shuffle=False, val_size=0.2, dataset_type="h5", return_idx=True):
    dataset_class = KITS_SLICE_h5 if(dataset_type=='h5') else KITS_SLICE_sim
    dataset = dataset_class(data_dir, return_idx=return_idx)
    train_size = int((1-val_size)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    return train_loader, test_loader


if __name__ == "__main__":
    data_dir = "/opt/data/private/why/dataset/KITS2019_preprocess/"
    # data_dir = "D:\dataset\KITS2019_preprocess"
    # train_iter, test_iter = get_loader(data_dir)
    # for img, seg in train_iter:
    #     print(1)
    # print("kits_loader test")
    # dataset = KITS_SLICE_sim(data_dir, return_idx=True)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for data in loader:
    #     sim = data[2].cpu().numpy()[0,:,:,0]
    #     cv_show(sim)
    #     sim = data[2].cpu().numpy()[0,:,:,1]
    #     cv_show(sim)
    train_iter, test_iter = get_loader(data_dir, dataset_type='sim', return_idx=False)
    for img, seg in train_iter:
        print(1)

