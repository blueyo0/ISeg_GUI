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
from data import KITS_SLICE_h5
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

class KITS_SLICE_sim(Dataset):
    def __init__(self, data_dir, return_idx=False):
        # self.case_num = 300
        self.data_dir = Path(data_dir)
        self.return_idx = return_idx
        self.size_arr = np.load(self.data_dir/"kits_slice_size.npy")
        self.case_num = self.size_arr.shape[0] 
        self.transform = transforms.Resize([512, 512])        

    def __getitem__(self, index):
        # TO-DO 二分根据Index确定case_id
        # print("data index:", index)
        lb , hb = 0, 0 # 下界，上界
        for case_id in range(self.case_num):
            hb = self.size_arr[case_id, 1]
            if(index>=lb and index<hb): break
            lb = hb

        slice_id = index - lb
        filename = self.data_dir / "kits_data_case_{:05d}.h5".format(case_id)
        h5_file = tables.open_file(filename, mode='r')
        img, seg = h5_file.root.img[slice_id].astype(np.float32), \
                   h5_file.root.seg[slice_id].astype(np.float32)
        sim = h5_file.root.sim[slice_id].astype(np.float32)
        # img = self.transform(img)
        # 数据检查
        # assert len(np.unique(seg)) == 2
        # assert not np.isnan(img).any()
        result = (img, seg, sim)
        if(self.return_idx):
            case_idx = h5_file.root.case[slice_id]
            slice_idx = h5_file.root.slice[slice_id]
            result = (img, seg, sim, np.array([case_idx, slice_idx]))
        h5_file.close()
        return result

    def __len__(self):
        return self.size_arr[-1,-1]



def get_loader(data_dir, shuffle=False, val_size=0.25):
    dataset = KITS_SLICE_h5(data_dir, return_idx=True)
    train_size = int((1-val_size)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    return train_loader, test_loader


if __name__ == "__main__":
    # data_dir = "/opt/data/private/why/dataset/KITS2019_modified/"
    data_dir = "D:\dataset\KITS2019_preprocess"
    # train_iter, test_iter = get_loader(data_dir)
    # for img, seg in train_iter:
    #     print(1)
    # print("kits_loader test")
    dataset = KITS_SLICE_sim(data_dir, return_idx=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        sim = data[2].cpu().numpy()[0,:,:,0]
        cv_show(sim)
        sim = data[2].cpu().numpy()[0,:,:,1]
        cv_show(sim)