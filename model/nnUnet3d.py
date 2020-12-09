# -*- encoding: utf-8 -*-
'''
@File    :   nnUnet3d.py
@Time    :   2020/12/08 14:06:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Iseensee的nnUnet的pytorch实现
'''

import torch
import torch.nn as nn

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

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dropout=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, \
                      kernel_size=kernel_size, stride=stride, \
                      padding=padding, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            
            nn.Conv3d(out_ch, out_ch, \
                      kernel_size=kernel_size, stride=stride, \
                      padding=padding, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
        )
        self.dropout = nn.Dropout3d(0.5) if(dropout) else None 

    def forward(self, x):
        x1 = self.conv(x)
        if(self.dropout): 
            x1 = self.dropout(x1)
        return x1

class upcat_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=True):
        super(upcat_conv_block, self).__init__()
        self.conv = conv_block(in_ch, out_ch, dropout=dropout).cuda()
        self.upconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(2,2,1), stride=(2,2,1))
    
    def forward(self, x_down, x):
        x_up = self.upconv(x_down)
        crop_idx = (x.size(2)-x_up.size(2))//2
        x_crop = x[:, :, crop_idx:x_up.size(2)+crop_idx, 
                         crop_idx:x_up.size(2)+crop_idx, 
                         crop_idx:x_up.size(2)+crop_idx]
        x_in = torch.cat((x_crop, x_up), 1)
        x_out = self.conv(x_in)
        return x_out

class nnUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, filter=15, dropout=True, device='cuda'):
        super(nnUnet3d, self).__init__()
        self.encode_filters = [in_ch, filter, filter*2, filter*4, filter*8, filter*16]
        self.decode_filters = [filter*16, filter*8, filter*4, filter*2, filter]
        self.convs = [conv_block(self.encode_filters[i], self.encode_filters[i+1], dropout=dropout)\
                      .to(device) for i in range(5)]
        self.pools = [nn.MaxPool3d(kernel_size=(2,2,1)) for i in range(4)]
        self.ups   = [upcat_conv_block(self.decode_filters[i], self.decode_filters[i+1], dropout=dropout)\
                      .to(device) for i in range(4)]
        self.final = nn.Sequential(nn.Conv3d(filter, out_ch, kernel_size=1),
                                   nn.Softmax(3))


    def forward(self, x):
        fm0 = self.convs[0](x)
        p0 = self.pools[0](fm0)
        fm1 = self.convs[1](p0)
        p1 = self.pools[1](fm1)
        fm2 = self.convs[2](p1)
        p2 = self.pools[2](fm2)
        fm3 = self.convs[3](p2)
        p3 = self.pools[3](fm3)
        fm4 = self.convs[4](p3)
        
        up3 = self.ups[0](fm4, fm3)
        up2 = self.ups[1](up3, fm2)
        up1 = self.ups[2](up2, fm1)
        up0 = self.ups[3](up1, fm0)

        out = self.final(up0)
        return out

        


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    x = torch.tensor(np.zeros([2, 1, 512, 512, 5]), dtype=torch.float32).cuda()
    net = nnUnet3d(1, 2, dropout=True).cuda()
    # net = nn.DataParallel(net, device_ids=[0,1])
    net.eval()
    
    print("start")
    with torch.no_grad():
        out1 = net(x)
        out2 = net(x)
        print(np.all(out1.cpu().numpy() == out2.cpu().numpy()))
    pass

