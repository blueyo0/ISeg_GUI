# -*- encoding: utf-8 -*-
'''
@File    :   simulate.py
@Time    :   2020/11/09 11:53:16
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   function of simulating user-interaction
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








