import nibabel as nib
import os
import torch
from torch.utils.data import DataLoader, Dataset

class BraTS_SLICE(Dataset):
    def __init__(self, data_dir, slice_no=64):
        self.data_num = 259 #TO-DO 全为HGG图像，实际数据集又369
        self.data_dir = data_dir
        self.slice_no = slice_no
        pass
    def __getitem__(self, index):
        dirname = os.path.join(self.data_dir, "BraTS20_Training_{:03d}".format(index+1))
        filename = os.path.join(dirname, "BraTS20_Training_{:03d}_flair.nii.gz".format(index+1))
        segname = os.path.join(dirname, "BraTS20_Training_{:03d}_seg.nii.gz".format(index+1))
        img, seg = nib.load(filename).get_fdata()[:,:,self.slice_no],\
                   nib.load(segname).get_fdata()[:,:,self.slice_no]
        img, seg =  torch.from_numpy(img).unsqueeze(0), torch.from_numpy(seg).unsqueeze(0)
        return img, seg
    def __len__(self):
        return self.data_num-1



# if __name__ == '__main__':
#     DATA_DIR = "D:\\dataset\\BraTS2020\\MICCAI_BraTS2020_TrainingData"
#     data = BraTS_SLICE(DATA_DIR)
#     loader = DataLoader(data, batch_size=5, shuffle=True)
#     for image, seg in loader:
#         image = image.unsqueeze(1)
#         print(image.shape)



