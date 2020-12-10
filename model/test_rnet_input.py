# -*- encoding: utf-8 -*-
'''
@File    :   test_rnet_input.py
@Time    :   2020/12/09 15:07:03
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   测试rnet输出的npz文件，查看问题
'''

import torch
import numpy as np
import random

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

from util.test import cv_show
from model.model_util import load_net_model

if __name__ == "__main__":
    
    map_0 = np.load("./model/input_save/map_0.npz")['arr_0']
    map_1 = np.load("./model/input_save/map_1.npz")['arr_0']

    idx = 7
    show_idx = 4
    arr = np.load("./model/input_save/img_{:05d}.npz".format(idx))['arr_0'][0:1, ...]
    p_0, p_1 = arr[0,3,...], arr[0,4,...]
    
    cv_show(map_0)
    cv_show(p_0)
    cv_show(map_1)
    cv_show(p_1)


    gt = np.load("./model/input_save/gt_{:05d}.npz".format(idx))['arr_0'][0:1, ...]
    # TO-DO: 文件读取
    cv_show(gt[0,1,...])
    arr_in = torch.from_numpy(arr)
    model = load_net_model("rnet")
    pred = model(arr_in).detach().cpu().numpy()
    cv_show(pred[0,1,...])
    pass




