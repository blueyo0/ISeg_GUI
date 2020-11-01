# -*- encoding: utf-8 -*-
'''
@File    :   train_kits.py
@Time    :   2020/10/30 10:24:11
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   使用KITS_2d的数据训练网络，获得分割
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

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

import sys
sys.path.append(os.getcwd())

from util.kits_loader import get_loader
from model.Unet import Unet

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def validation(model, loader_val):
    '''
    :param model: 分割模型
    :param loader_val: 测试数据的Dataloder
    :return: slice-level dice, case-level dice
    '''
    val_dice_slice = {}
    val_dice_case = {}

    # 计算 slice_level 的 dice coefficient
    for i, batch in enumerate(loader_val):
        if(i>20): break #TO-DO
        slice_name, img, gt = batch[2], batch[0].cuda(), batch[1].cuda()
        if(img.shape[2]!=512 or img.shape[3]!=512):
            print("error")
        with torch.no_grad():
            model.eval()
            pred = model(img)
        slice_name = slice_name[0,:,0]
        val_dice_slice[slice_name] = 1 - dice_loss((pred[:, 1, :, :] >= 0.5).float(), gt[:, 1, :, :].float()).item()

    # 计算 case_level 的 dice_coefficient
    key_arr = np.array([v.numpy() for v in list(val_dice_slice.keys())])
    case_names = np.unique(key_arr[:,0])
    for case_name in case_names:
        val_dice_case[case_name] = []  # 每个case初始化为空列表
    for slice_name, slice_dice in val_dice_slice.items():
        val_dice_case[int(slice_name[0].item())].append(slice_dice)  # 每个case的slice dice添加入列表

    return np.mean([np.mean(v) for v in val_dice_case.values()])

if __name__ == "__main__":
    DATA_DIR = "/opt/data/private/why/dataset/KITS2019_modified/"
    MODEL_DIR = "/opt/data/private/why/model/KITS2019"

    train_loader, valid_loader = get_loader(DATA_DIR)
    net_params = {"num_classes" : 2, "num_channels" : 3, "num_filters" : 32}
    train_params = {    
        "iterations"     : 30000,
        "learning_rate"  : 1e-3,
        "momentum"       : 0.9,
        "weight_decay"   : 1e-8,
        # "print_freq"     : 10,
        # "val_freq"       : 10,
        "print_freq"     : 200,
        "val_freq"       : 1000,
        "save_freq"      : 2000,
        "lr_decay_freq"  : 500
    }
    model = Unet(net_params=net_params)
    # model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), 
                          lr=train_params['learning_rate'], 
                          momentum=train_params['momentum'],
                          weight_decay=train_params['weight_decay'])
    
    bce_criterion = torch.nn.BCELoss()
    dice_criterion = dice_loss

    loader_iter = iter(train_loader)
    global_step = 0
    best_cur_dice = 0
    lr_ = train_params['learning_rate']

    while global_step < train_params['iterations']:
        try:
            batch = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        model.train()

        img, gt = batch[0].cuda(), batch[1].cuda()
        pred = model(img)
        # print(np.max(pred.cpu().detach().numpy()), np.min(pred.cpu().detach().numpy()))
        loss_bce = bce_criterion(pred, gt)
        loss_dice = dice_criterion(pred[:, 1, :, :], gt[:, 1, :, :])
        loss = 0.5 * loss_bce + 0.5 * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # 打印训练结果
        # TO-DO: 修改一下输出部分
        if global_step % train_params['print_freq'] == 0:
            print('Iteration %d (lr-%.4f): loss_bce : %.4f; loss_dice: %.4f; loss: %.4f; ' %
                    (global_step, optimizer.param_groups[0]['lr'], loss_bce.item(), loss_dice.item(), loss.item()))

        if global_step % train_params['val_freq'] == 0:
            cur_dice = validation(model, valid_loader)
            print("Valid dice: %.4f" % cur_dice)

            # 保存最优模型
            if cur_dice >= best_cur_dice:
                ckpt_path = os.path.join(MODEL_DIR, 'best_model.pth')
                torch.save(model.state_dict(), ckpt_path)
                print('------> cur_dice improved from %.4f to %.4f' % (best_cur_dice, cur_dice))
                best_cur_dice = cur_dice
            else:
                print('------> val_dice did not improved from %.4f' % (best_cur_dice))

            # 定期保存模型
            if global_step % train_params['save_freq'] == 0:
                ckpt_path = os.path.join(MODEL_DIR,
                                         'step_%d_curdice_%.4f.pth' %
                                         (global_step, cur_dice))
                torch.save(model.state_dict(), ckpt_path)
                print('Iteration %d: save model to %s' % (global_step, ckpt_path))

            if global_step % train_params['lr_decay_freq'] == 0:
                lr_ = train_params['learning_rate'] * 0.95 ** (global_step // train_params['lr_decay_freq'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

    print("Training finished!")

