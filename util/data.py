import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset

class BraTS_SLICE(Dataset):
    def __init__(self, data_dir, slice_no=64):
        self.data_num = 259 #TO-DO 全为HGG图像，实际数据集大小369
        self.data_dir = data_dir
        self.slice_no = slice_no
        pass
    def __getitem__(self, index):
        dirname = os.path.join(self.data_dir, "BraTS20_Training_{:03d}".format(index+1))
        filename = os.path.join(dirname, "BraTS20_Training_{:03d}_flair.nii.gz".format(index+1))
        segname = os.path.join(dirname, "BraTS20_Training_{:03d}_seg.nii.gz".format(index+1))
        img, seg = nib.load(filename).get_fdata()[:,:,self.slice_no],\
                   nib.load(segname).get_fdata()[:,:,self.slice_no]
        
        # 去掉所有4以下的
        for x in np.nditer(seg, op_flags=['readwrite']):
            x[...]= 1 if(x>0) else 0

        img, seg =  torch.from_numpy(img).float().unsqueeze(0),\
                    torch.from_numpy(seg).float().unsqueeze(0)
        return img, seg
    def __len__(self):
        return self.data_num-1


