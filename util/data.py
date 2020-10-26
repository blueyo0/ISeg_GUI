import nibabel as nib
import numpy as np
import tables
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

class BraTS_SLICE_h5(Dataset):
    def __init__(self, h5_file):
        self.data_num = 259 #TO-DO 全为HGG图像，实际数据集大小369
        self.h5_file = tables.open_file(h5_file, mode='r')
    def __getitem__(self, index):
        img, seg = self.h5_file.root.img[index], self.h5_file.root.seg[index]
        img, seg =  torch.from_numpy(img).float().unsqueeze(0),\
                    torch.from_numpy(seg).float().unsqueeze(0)
        return img, seg
    def __len__(self):
        return self.data_num-1
        


def create_data_file(out_file, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    image_shape = tuple([0] + list(image_shape))
    img_storage = hdf5_file.create_earray(hdf5_file.root, 'img', tables.Float32Atom(), shape=image_shape,
                                           filters=filters, expectedrows=n_samples)
    seg_storage = hdf5_file.create_earray(hdf5_file.root, 'seg', tables.UInt8Atom(), shape=image_shape,
                                            filters=filters, expectedrows=n_samples)
    return hdf5_file, img_storage, seg_storage

def preprocess_data_to_hdf5(data_dir, out_file, image_shape, n_samples):
    # try:
    hdf5_file, img_storage, seg_storage = create_data_file(out_file,
                                                            n_samples=n_samples,
                                                            image_shape=image_shape)
    # except Exception as e:
    #     # If something goes wrong, delete the incomplete data file
    #     os.remove(out_file)
    #     raise e
    
    data = BraTS_SLICE(data_dir)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    for img, seg in loader:
        img_storage.append(img.cpu().detach().numpy()[0,0,:,:][np.newaxis])
        seg_storage.append(seg.cpu().detach().numpy()[0,0,:,:][np.newaxis])

    hdf5_file.close()
    return out_file

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)