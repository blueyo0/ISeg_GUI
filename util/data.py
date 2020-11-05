# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2020/10/29 08:58:06
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
'''

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import tables
import os
import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from imageio import imwrite
from PIL import Image
import platform

# from numba import jit
class KITS_SLICE_h5(Dataset):
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
        # img = self.transform(img)
        # 数据检查
        # assert len(np.unique(seg)) == 2
        # assert not np.isnan(img).any()
        result = (img, seg)
        if(self.return_idx):
            case_idx = h5_file.root.case[slice_id]
            slice_idx = h5_file.root.slice[slice_id]
            result = (img, seg, np.array([case_idx, slice_idx]))
        h5_file.close()
        return result

    def __len__(self):
        return self.size_arr[-1,-1]

class BraTS_SLICE(Dataset):
    def __init__(self, data_dir, slice_no=64):
        self.data_num = 259 
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
        self.data_num = 259 
        self.h5_file = tables.open_file(h5_file, mode='r')
    def __getitem__(self, index):
        img, seg = self.h5_file.root.img[index], self.h5_file.root.seg[index]
        img, seg =  torch.from_numpy(img).float().unsqueeze(0),\
                    torch.from_numpy(seg).float().unsqueeze(0)
        return img, seg
    def __len__(self):
        return self.data_num-1
        


def create_data_file(out_file, n_samples, image_shape, mask_shape=None, use_idx=False):
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
    if(not use_idx):    
        return hdf5_file, img_storage, seg_storage
    else: 
        caseIdx_storage = hdf5_file.create_earray(
                        hdf5_file.root, 'case', tables.UInt8Atom(), 
                        shape=(0, 1), filters=filters, expectedrows=n_samples)
        sliceIdx_storage = hdf5_file.create_earray(
                        hdf5_file.root, 'slice', tables.UInt8Atom(), 
                        shape=(0, 1), filters=filters, expectedrows=n_samples)
        return hdf5_file, img_storage, seg_storage, caseIdx_storage, sliceIdx_storage  

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
    sysstr = platform.system()
    if(sysstr=="Windows"):
        data_path = Path("D:/dataset/KITS2019/data")
    elif(sysstr=="Linux"):
        data_path = Path("/opt/data/private/why/dataset/KITS_2019/data")
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

def load_volume(cid, mode):
    case_path = get_case_path(cid)
    if(mode=='nib'):
        vol = nib.load(str(case_path / "imaging.nii.gz"))
    elif(mode=='sitk'):
        itk_vol = sitk.ReadImage(str(case_path / "imaging.nii.gz"))
        vol = sitk.GetArrayFromImage(itk_vol)#.transpose([1,2,0])
    return vol


def load_segmentation(cid, mode):
    case_path = get_case_path(cid)
    if(mode=='nib'):
        seg = nib.load(str(case_path / "segmentation.nii.gz"))
    elif(mode=='sitk'):
        itk_seg = sitk.ReadImage(str(case_path / "segmentation.nii.gz"))
        seg = sitk.GetArrayFromImage(itk_seg)#.transpose([1,2,0])
    return seg


def load_case(cid, mode='nib'):
    vol = load_volume(cid, mode)
    seg = load_segmentation(cid, mode)
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
        if(np.isnan(image_patch).any()): continue
        if(image_patch.shape[0]!=512 or image_patch.shape[1]!=512): continue
        mask_patch = mask[:,:,z_i]
        mask_patch = _label_decomp(mask_patch, 2)

        image_patches.append(image_patch.transpose([2,0,1]))
        mask_patches.append(mask_patch.transpose([2,0,1]))
        slice_indexs.append(z_i)
    return image_patches, mask_patches, slice_indexs

def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)

def arr2img(arr, position='a'):
    img = None
    if(position=='a'):
        img = Image.fromarray(np.uint8(arr)).rotate(90)\
                .transpose(Image.FLIP_TOP_BOTTOM).toqimage()
    else:
        img = Image.fromarray(np.uint8(arr)).toqimage()
    return img


# TO-DO 重写load_nii_data函数
def load_nii_data(path):
    itk_vol = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(itk_vol)

    assert vol.ndim==3
    if(vol.shape[0]==vol.shape[2]): 
        vol = vol.transpose([0,2,1])
    elif(vol.shape[1]==vol.shape[2]): 
        vol = vol.transpose([1,2,0])
        
    vol_max, vol_min = np.max(vol), np.min(vol)
    nvol = (vol-vol_min) / max(vol_max-vol_min, 1e-3)
    nvol = 255*nvol

    # img_2d_li_a = [arr2img(nvol[:,:,i_a]) for i_a in range(nvol.shape[2])]
    # img_2d_li_s = [arr2img(nvol[i_s,:,:], position='s') for i_s in range(nvol.shape[0])]
    # img_2d_li_c = [arr2img(nvol[:,i_c,:], position='c') for i_c in range(nvol.shape[1])]
    # return [img_2d_li_a, img_2d_li_s, img_2d_li_c]

    return nvol

if __name__ == "__main__":
    # 用KITS测试npz和h5间的性能对比
    # 测试结果，npz 显著慢于 h5
    '''
    SSD上
    h5 file-loading time cost:0.00300 s
    h5 data-reading time cost:0.04500 s
    npz file-loading time cost:0.00300 s
    npz data-reading time cost:31.98563 s
    服务器上
    h5 save time cost:6.4 s
    npz save time cost:20.4 s
    h5 file-loading time cost:0.06992 s
    h5 data-reading time cost:0.19449 s
    npz file-loading time cost:0.25221 s
    npz data-reading time cost:30.78306 s
    '''
    OVERWRITE = True#覆盖旧文件
    # MODE = "preprocess"
    # MODE = "test"
    MODE = "dataset"


    if(MODE=="test"):
        WRITE_TEST = True #写入测试
        READ_TEST = True #读取测试
        REAL_PROPRECESS = False #实际预处理代码
    elif(MODE=="preprocess"):
        WRITE_TEST = False #写入测试
        READ_TEST = False #读取测试
        REAL_PROPRECESS = True #实际预处理代码
    elif(MODE=="dataset"):
        WRITE_TEST = False #写入测试
        READ_TEST = False #读取测试
        REAL_PROPRECESS = False #实际预处理代码

    if(MODE=="dataset"):
        data_dir = Path("/opt/data/private/why/dataset/KITS2019_modified/")
        prev_dir = data_dir / "preview"
        dataset = KITS_SLICE_h5(data_dir, return_idx=True)
        if(not prev_dir.exists()): os.makedirs(prev_dir)

        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        for i, (img, seg, idx) in enumerate(loader):
            img_out = img[0,1,:,:].cpu().numpy()
            img_out = hu_to_grayscale(img_out, None ,None).astype(np.uint8)
            imwrite(str(prev_dir/"img_{:05d}.png".format(i*5)), img_out)
            seg_out = seg[0,1,:,:].cpu().numpy().astype(np.uint8)
            seg_out[seg_out==1] = 255
            # print(np.unique(seg_out))
            imwrite(str(prev_dir/"seg_{:05d}.png".format(i*5)), seg_out)
            # imwrite(str(prev_dir/"seg_{:05d}_1.png".format(i*5)), seg[0,:,:,1].cpu().numpy())
            idx = idx.cpu().numpy()
            for order in range(5):
                print("case %d, slice %d."%(idx[order,0], idx[order,1]))


    # 程序入口
    if(WRITE_TEST or READ_TEST):
        # output_train_dir = Path("D:/dataset/KITS2019_modified")
        output_train_dir = Path("/opt/data/private/why/dataset/KITS2019_modified")
        if(not output_train_dir.exists()): os.makedirs(output_train_dir)
        out_file = output_train_dir / "kits_data.h5"
        out_file2 = output_train_dir / "kits_data.npz"

    if(WRITE_TEST):
        img, seg = load_case(2)
        # img = img.get_fdata() 
        # seg = seg.get_fdata()
        img, seg = np.array(img.dataobj).transpose([1,2,0]), \
                    np.array(seg.dataobj).transpose([1,2,0])
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
                # print(hdf5_file.get_filesize())
                img_storage.append(img_patches[i][np.newaxis])
                seg_storage.append(seg_patches[i][np.newaxis])

                # print(hdf5_file.get_filesize())
            hdf5_file.close()
            print("h5 save time cost:%.1f s" % (time.time()-start))
        # npz存储
        # if((not os.path.exists(out_file2)) or OVERWRITE):
        #     start = time.time()
        #     np.savez(out_file2, img_patches, seg_patches)
        #     print("npz save time cost:%.1f s" % (time.time()-start))


    if(READ_TEST):
        start = time.time()
        h5_file = tables.open_file(out_file, mode='r')
        print("h5 file-loading time cost:%.5f s" % (time.time()-start))

        start = time.time()
        for i in range(10):
            img1, seg2 = h5_file.root.img[i], h5_file.root.seg[i]
        print("h5 data-reading time cost:%.5f s" % ((time.time()-start)))


        # start = time.time()
        # np_file = np.load(out_file2)
        # print("npz file-loading time cost:%.5f s" % (time.time()-start))

        # start = time.time()
        # for i in range(10):
        #     img2, seg2 = np_file['arr_0'][i], np_file['arr_1'][i]
        # print("npz data-reading time cost:%.5f s" % (time.time()-start))


    if(REAL_PROPRECESS):
        pre_data_dir = Path("/opt/data/private/why/dataset/KITS2019_modified/")
        if(not pre_data_dir.exists()): os.makedirs(pre_data_dir)
        
        print("Data preprocessing started.")
        start = time.time() 
        slice_num_arr = []
        slice_sum = 0
        slice_size_file = pre_data_dir/"kits_slice_size.npy"
        for i in range(200):
            out_file =  pre_data_dir / ("kits_data_"+get_full_case_id(i)+".h5")
            img, seg = load_case(i, mode='sitk')
            print("[%03d]Data file is loaded"%(i), end="\r")
            mean = np.mean(img)
            std = np.std(img)
            normalized_img = (img - mean) / std  # 正则化处理   
            print("[%03d]Data normalization finished"%(i), end="\r")

            img_patches, seg_patches, slice_indexs = _extract_patch(normalized_img, seg)
            slice_sum += len(slice_indexs)
            slice_num_arr.append([len(slice_indexs), slice_sum])
            np.save(slice_size_file, np.array(slice_num_arr))


            if((not os.path.exists(out_file)) or OVERWRITE):
                hdf5_file, img_storage, seg_storage, \
                caseIdx_storage, sliceIdx_storage = create_data_file(
                                                out_file, n_samples=img.shape[2], 
                                                image_shape=(3, img.shape[0], img.shape[1]),
                                                mask_shape=(2, seg.shape[0], seg.shape[1]),
                                                use_idx=True)

                for idx, slice_idx in enumerate(slice_indexs):
                    img_storage.append(img_patches[idx][np.newaxis])
                    seg_storage.append(seg_patches[idx][np.newaxis])
                    caseIdx_storage.append(np.array([i])[np.newaxis])
                    sliceIdx_storage.append(np.array([slice_idx])[np.newaxis])
                    print("[%03d]Slice %3d is compressed"%(i, slice_idx), end="\r")

                hdf5_file.close()
            total_sec = (time.time()-start)
            h, m, s = total_sec/3600, (total_sec%3600)/60, total_sec%60
            pred_sec = (300-i-1)/(i+1)*total_sec
            pred_h, pred_m, pred_s = pred_sec/3600, (pred_sec%3600)/60, pred_sec%60
            print("Case {:03d} Elapsed Time: {:02d}:{:02d}:{:02d} \t Left Time: {:02d}:{:02d}:{:02d}"\
                .format(i,int(h), int(m), int(s),
                          int(pred_h), int(pred_m), int(pred_s)))
              
        # np.save(slice_size_file, slice_num_arr)
        size_arr = np.load(slice_size_file)
        print(size_arr)
        print("Data preprocessing finished!")
