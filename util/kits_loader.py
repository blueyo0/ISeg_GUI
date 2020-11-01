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
import sys
import os
sys.path.append(os.getcwd())
from util.data import KITS_SLICE_h5

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

def get_loader(data_dir, shuffle=False, val_size=0.25):
    dataset = KITS_SLICE_h5(data_dir, return_idx=True)
    train_size = int((1-val_size)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    return train_loader, test_loader


if __name__ == "__main__":
    data_dir = "/opt/data/private/why/dataset/KITS2019_modified/"
    train_iter, test_iter = get_loader(data_dir)
    for img, seg in train_iter:
        print(1)
    print("kits_loader test")