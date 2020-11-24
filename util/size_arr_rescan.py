# -*- encoding: utf-8 -*-

'''
@File    :   size_arr_rescan.py
@Time    :   2020/11/24 21:29:51
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   重新扫描h5文件，确认size_arr文件正确
'''

import glob
import tables
import numpy as np
from pathlib import Path

data_dir = "/opt/data/private/why/dataset/KITS2019_preprocess/"
slice_size_file = Path(data_dir)/"kits_size_refine.npy"
file_li = glob.glob(data_dir+"*h5")
total_slice_num = 0
slice_num_arr = []
for filename in file_li:
    h5_file = tables.open_file(filename, mode='r')
    size = h5_file.root.img.shape[0]
    total_slice_num += size
    slice_num_arr.append([size, total_slice_num])

np.save(slice_size_file, np.array(slice_num_arr))


