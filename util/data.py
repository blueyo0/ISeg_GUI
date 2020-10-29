# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2020/10/29 08:58:06
@Author  :   Haoyu Wang 
@Version :   1.0
@Contact :   small_dark@sina.com
'''

import nibabel as nib
import numpy as np
import tables
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


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
        


def create_data_file(out_file, n_samples, image_shape, mask_shape=None):
    '''
    :out_file: 输出文件路径
    :n_sampels: 预计的img数量
    '''
    mask_shape = image_shape if(not mask_shape) else mask_shape
    image_shape = tuple([0] + list(image_shape))
    mask_shape = tuple([0] + list(mask_shape))
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    img_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'img', tables.Float32Atom(), 
                    shape=image_shape, filters=filters, expectedrows=n_samples)
    seg_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'seg', tables.UInt8Atom(), 
                    shape=mask_shape, filters=filters, expectedrows=n_samples)
    return hdf5_file, img_storage, seg_storage

def preprocess_data_to_hdf5(data_dir, out_file, image_shape, n_samples):
    # try:
    hdf5_file, img_storage, seg_storage = create_data_file(
                                            out_file, n_samples=n_samples, 
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

# KITS读取代码

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path("D:/dataset/KITS2019/data")
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path


def load_volume(cid):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    return vol


def load_segmentation(cid):
    case_path = get_case_path(cid)
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return seg


def load_case(cid):
    vol = load_volume(cid)
    seg = load_segmentation(cid)
    return vol, seg


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
    for i in range(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)
    return np.array(one_hot).transpose([1,2,0])


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



if __name__ == "__main__":
    # 用KITS测试npz和h5间的性能对比
    # 测试结果，npz 显著慢于 h5
    '''
    h5 file-loading time cost:0.00300 s
    h5 data-reading time cost:0.04500 s
    npz file-loading time cost:0.00300 s
    npz data-reading time cost:31.98563 s
    '''
    OVERWRITE = False
    WRITE_TEST = False
    READ_TEST = True

    output_train_dir = Path("D:/dataset/KITS2019_modified")
    if(not output_train_dir.exists()): os.makedirs(output_train_dir)
    out_file = output_train_dir / "kits_data.h5"
    out_file2 = output_train_dir / "kits_data.npz"

    if(WRITE_TEST):
        img, seg = load_case(2)
        img = img.get_fdata() 
        seg = seg.get_fdata()
        img, seg = np.array(img).transpose([1,2,0]), \
                    np.array(seg).transpose([1,2,0])
        mean = np.mean(img)
        std = np.std(img)
        normalized_img = (img - mean) / std  # 正则化处理   

        img_patches, seg_patches, slice_indexs = _extract_patch(normalized_img, seg)

        # h5存储
        if((not os.path.exists(out_file)) or OVERWRITE):
            start = time.time() 
            hdf5_file, img_storage, seg_storage = create_data_file(
                                                    out_file, n_samples=img.shape[2], 
                                                    image_shape=(img.shape[0], img.shape[1], 3),
                                                    mask_shape=(img.shape[0], img.shape[1], 2))
            for i, slice_idx in enumerate(slice_indexs):
                img_storage.append(img_patches[i][np.newaxis])
                seg_storage.append(seg_patches[i][np.newaxis])

            hdf5_file.close()
            print("h5 save time cost:%.1f s" % (time.time()-start))
        # npz存储
        if((not os.path.exists(out_file2)) or OVERWRITE):
            start = time.time()
            np.savez(out_file2, img_patches, seg_patches)
            print("npz save time cost:%.1f s" % (time.time()-start))


    if(READ_TEST):
        start = time.time()
        h5_file = tables.open_file(out_file, mode='r')
        print("h5 file-loading time cost:%.5f s" % (time.time()-start))

        start = time.time()
        for i in range(10):
            img1, seg2 = h5_file.root.img[i], h5_file.root.seg[i]
        print("h5 data-reading time cost:%.5f s" % ((time.time()-start)))


        start = time.time()
        np_file = np.load(out_file2)
        print("npz file-loading time cost:%.5f s" % (time.time()-start))

        start = time.time()
        for i in range(10):
            img2, seg2 = np_file['arr_0'][i], np_file['arr_1'][i]
        print("npz data-reading time cost:%.5f s" % (time.time()-start))