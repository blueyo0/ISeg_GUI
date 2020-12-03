# -*- encoding: utf-8 -*-
'''
@File    :   test_kits.py
@Time    :   2020/11/01 13:38:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   读取train中的模型在KITS上实际测试效果
'''

import os
import sys
sys.path.append(os.getcwd())

from model.Unet import Unet
from util.data import load_case, hu_to_grayscale
from util.data import _patch_center_z, _label_decomp
from imageio import imwrite
from pathlib import Path

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
torch.set_default_tensor_type('torch.DoubleTensor')

if __name__ == '__main__':
    # MODEL_PATH =  "D:/BME/model/best_model.pth"
    MODEL_PATH =  "./model/prnet/pnet_model.pth"
    prev_dir = Path(os.path.join("D:/BME/model/", "preview_pnet"))
    
    # MODEL_PATH =  "D:/BME/model/step_22000_curdice_0.1421.pth"
    # prev_dir = Path(os.path.join("D:/BME/model/", "preview_2"))

    net_params = {'num_filters':32, 'num_channels':3, 'num_classes':2}
    model = Unet(net_params)
    model.load_state_dict(torch.load(MODEL_PATH))

    # caseId = 1
    caseRange = [1, 10]
    for caseId in range(caseRange[0], caseRange[1]):
        img, seg = load_case(caseId, mode='sitk')
        z = _patch_center_z(seg)[10].item()
        img_patch = img[:,:,z-1:z+2]
        for i in range(3):
            img_single = img_patch[:,:,i]
            mean = np.mean(img_single)
            std = np.std(img_single)
            normalized_img = (img_single - mean) / std  # 正则化处理  
            img_patch[:,:,i] = normalized_img
        img_patch = torch.from_numpy(img_patch.transpose([2,0,1])[np.newaxis])
        # seg_patch = _label_decomp(seg, 2)
        pred = model(img_patch)

        img_out = img_patch[0,1,:,:].cpu().numpy()
        img_out = hu_to_grayscale(img_out, None ,None).astype(np.uint8)
        imwrite(str(prev_dir/"img_{:05d}.png".format(caseId)), img_out)

        seg_out = seg[:,:,z].astype(np.uint8)
        seg_out[seg_out==1] = 255
        imwrite(str(prev_dir/"seg_{:05d}.png".format(caseId)), seg_out)

        pred_out = pred[0,1,:,:].detach().cpu().numpy()
        pred_out[pred_out>0.5] = 255
        pred_out = pred_out.astype(np.uint8)
        imwrite(str(prev_dir/"pred_{:05d}.png".format(caseId)), pred_out)
        print("caseId %3d is tested" % caseId)

    print("endl")
    # img_out = img[0,1,:,:].cpu().numpy()
    # img_out = hu_to_grayscale(img_out, None ,None).astype(np.uint8)
    # imwrite(str(prev_dir/"img_{:05d}.png".format(i*5)), img_out)
    # seg_out = seg[0,1,:,:].cpu().numpy().astype(np.uint8)
    # seg_out[seg_out==1] = 255
    # # print(np.unique(seg_out))
    # imwrite(str(prev_dir/"seg_{:05d}.png".format(i*5)), seg_out)

