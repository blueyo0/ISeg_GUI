# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2020/11/21 15:04:09
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   执行完整的数据预处理
'''

import argparse
import os
import platform
import time
import tables
import SimpleITK as sitk
import numpy as np
import cv2
from pathlib import Path
from data import load_case_sitk, extract_kits_patch, get_full_case_id, grayscale2bgra
from simulate import randomSample, randomScribble, getEuclidDistanceMap
from test import cv_show, cv_show_with_sim




def create_h5_file(out_file, n_samples, img_shape, sim_shape, seg_shape):
    '''
    :out_file: 输出文件路径
    :n_sampels: 预计的img数量
    '''
    img_shape = tuple([0] + list(img_shape))
    sim_shape = tuple([0] + list(sim_shape))
    seg_shape = tuple([0] + list(seg_shape))
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    img_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'img', tables.Float32Atom(), 
                    shape=img_shape, filters=filters, expectedrows=n_samples)
    sim_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'sim', tables.Float32Atom(), 
                    shape=sim_shape, filters=filters, expectedrows=n_samples)
    seg_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'seg', tables.UInt8Atom(), 
                    shape=seg_shape, filters=filters, expectedrows=n_samples)
    caseIdx_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'case', tables.UInt8Atom(), 
                    shape=(0, 1), filters=filters, expectedrows=n_samples)
    sliceIdx_storage = hdf5_file.create_earray(
                    hdf5_file.root, 'slice', tables.UInt8Atom(), 
                    shape=(0, 1), filters=filters, expectedrows=n_samples)
    return hdf5_file, img_storage, sim_storage, seg_storage, caseIdx_storage, sliceIdx_storage  

def getTime(start, curr_i, total_i):
    total_sec = (time.time()-start)
    h, m, s = total_sec/3600, (total_sec%3600)/60, total_sec%60
    pred_sec = (total_i-curr_i-1)/(curr_i+1)*total_sec
    pred_h, pred_m, pred_s = pred_sec/3600, (pred_sec%3600)/60, pred_sec%60
    return h, m, s, pred_h, pred_m, pred_s

def getSimInteraction(img_patches, seg_patches):
    img, seg = img_patches[:,:,1], seg_patches[:,:,1]
    seg = grayscale2bgra(seg)
    # negative interaction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    dilated_seg = cv2.dilate(seg, kernel, iterations=3)
    nega = cv2.subtract(dilated_seg, seg)
    nega[:,:,3] = 255

    li_0 = randomSample(nega, pt_step=25)
    
    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated_nega = cv2.dilate(nega, new_kernel, iterations=1)
    # li_0 = randomScribble(li_0, dilated_nega)

    # dilated_nega = cv2.dilate(nega, kernel, iterations=3)
    dist_map_0 = getEuclidDistanceMap(li_0, nega)

    # if(len(li_0)>0):
    #     cv_show_with_sim(dist_map_0, li_0=li_0)

    # positive interaction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    posi = cv2.erode(seg, kernel, iterations=3)
    li_1 = randomSample(posi)
    # TO-DO： 先不生成scribble
    # li_1 = randomScribble(li_1, seg)
    dist_map_1 = getEuclidDistanceMap(li_1, seg)
    # cv_show_with_sim(dist_map, li_1=li_1)
    dist_map_0 = cv2.cvtColor(dist_map_0, cv2.COLOR_BGRA2GRAY).astype(np.float)/255
    dist_map_1 = cv2.cvtColor(dist_map_1, cv2.COLOR_BGRA2GRAY).astype(np.float)/255
    sim = np.array([dist_map_0, dist_map_1]).transpose([1,2,0])
    return sim

if __name__ == "__main__":
    # 默认地址设置
    ini_data_dir =Path("/opt/data/private/why/dataset/KITS_2019/data")
    pre_data_dir = Path("/opt/data/private/why/dataset/KITS2019_preprocess/")
    if(platform.system()=='Windows'): 
        ini_data_dir = Path("D:/dataset/KITS2019/data")
        pre_data_dir = Path("D:/dataset/KITS2019_preprocess")
    if(not pre_data_dir.exists()): os.makedirs(pre_data_dir)

    # 初始化命令行参数
    parser = argparse.ArgumentParser(description="preprocess the dataset", allow_abbrev=False)
    parser.add_argument("-c", "--continue", action='store_false', default=True, dest='overwrite',help='set overwrite flag to False')
    parser.add_argument("-d", "--data_folder", default=str(ini_data_dir), type=str, help="training data folder")
    parser.add_argument("-r", "--result_folder", default=str(pre_data_dir), type=str, help="output data folder")
    parser.add_argument("-n", "--case_number", default=210, type=int, help="training case number")
    args = parser.parse_args()
    # print(args)

    # 预处理主要内容
    print("Data preprocessing started.")
    start = time.time() 

    slice_sum = 0
    slice_num_arr = []
    slice_size_file = Path(args.result_folder)/"kits_slice_size.npy"
    for i in range(args.case_number):
        img, seg = load_case_sitk(i, args.data_folder)
        
        h, m, s, pred_h, pred_m, pred_s = getTime(start, i, args.case_number)
        print("[%03d]Data file is loaded.   Elapsed Time: %02d:%02d:%02d                    "\
              %(i, h, m, s), end="\r")

        mean = np.mean(img)
        std = np.std(img)
        normalized_img = (img - mean) / std  # 正则化处理  

        h, m, s, pred_h, pred_m, pred_s = getTime(start, i, args.case_number)
        print("[%03d]Data normalization finished.   Elapsed Time: %02d:%02d:%02d             "\
              %(i, h, m, s), end="\r")

        
        img_patches, seg_patches, slice_indexs = extract_kits_patch(normalized_img, seg)
        slice_sum += len(slice_indexs)
        slice_num_arr.append([len(slice_indexs), slice_sum])
        h, m, s, pred_h, pred_m, pred_s = getTime(start, i, args.case_number)
        print("[%03d]Data label extracted.   Elapsed Time: %02d:%02d:%02d                    "\
              %(i, h, m, s), end="\r")
    
        out_file =  Path(args.result_folder) / ("kits_data_"+get_full_case_id(i)+".h5")
        if(args.overwrite or (not os.path.exists(out_file))):
            hdf5_file, img_storage, sim_storage, seg_storage, \
            caseIdx_storage, sliceIdx_storage = create_h5_file(
                out_file, n_samples=len(img_patches),
                img_shape=(512, 512, 3),
                sim_shape=(512, 512, 2),
                seg_shape=(512, 512, 2),
            )
            # count = 10
            for idx, slice_idx in enumerate(slice_indexs):
                img_storage.append(img_patches[idx][np.newaxis])
                seg_storage.append(seg_patches[idx][np.newaxis])
                caseIdx_storage.append(np.array([i])[np.newaxis])
                sliceIdx_storage.append(np.array([slice_idx])[np.newaxis])
                # TO-DO 生成Sim, 储存所有数据
                sim_patch = getSimInteraction(img_patches[idx], seg_patches[idx])
                sim_storage.append(sim_patch[np.newaxis])
                print("[%03d]Slice %3d is compressed                                         "%\
                      (i, slice_idx), end="\r")
                # count -= 1
                # if(count<0): break

            hdf5_file.close()
            pass
        
        np.save(slice_size_file, np.array(slice_num_arr))
        h, m, s, pred_h, pred_m, pred_s = getTime(start, i, args.case_number)
        print("[%03d]Data preprocessing finished.   Elapsed Time: %02d:%02d:%02d    Left Time:%02d:%02d:%02d"\
              %(i, h, m, s, pred_h, pred_m, pred_s), end="\n")
    pass





