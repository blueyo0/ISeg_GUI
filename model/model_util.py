# -*- encoding: utf-8 -*-
'''
@File    :   model_util.py
@Time    :   2020/12/02 20:22:44
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   模型加载，使用相关的一些API
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

from model.Unet import Unet

def load_net_model(type='pnet', net_params={'num_filters':32, 
                                            'num_channels':3, 
                                            'num_classes':2}):
    model = Unet(net_params)
    model.load_state_dict(torch.load("./model/prnet/pnet_model.pth"))
    return model

def predict(model, img_patch):
    img_patch = torch.from_numpy(img_patch.transpose([2,0,1])[np.newaxis].astype(np.float32))
    # img_patch = img_patch.double()
    pred = model(img_patch)
    pred_out = pred[0,1,:,:].detach().cpu().numpy()
    pred_out[pred_out>0.5] = 255
    pred_out = pred_out.astype(np.uint8)
    return pred_out