#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 3:17 下午
# @Author  : Jingyang.Zhang
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
'''
根据每个中心的数据单独进行训练; 注意此v1版本：模型初始化、优化器初始化在 single_train_round 内部
'''
import os, random, torch, shutil, logging
from settings import Settings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

### 实验重复性
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')
###
from lib.separatedataset import SeparateDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from lib.unet import Unet
import torch.optim as optim
import torch.nn as nn
from lib.losses import dice_loss
import numpy as np
from lib.validation import validation


def inference(model, loaders_val:dict):
    '''
    在所有中心的数据上进行validation，返回 dice={'ISBI':0.84, 'ISBI_1.5T':0.81, ..}
    '''
    dice = {site: validation(model, loader_val) for site, loader_val in loaders_val.items()}
    return dice


def single_train_round(site:str, loader_train:DataLoader, loaders_val:dict,
                       net_params:dict, train_params:dict, snap_path):
    '''
    在单个中心上进行模型训练，然后在所有中心的数据上测试
    :param loader_train: 当前中心数据的Dataloader
    :param loaders_val: 所有中心数据的Dataloader的字典，e.g. {'ISBI':Dataloader, 'ISBI_1.5T':Dataloader}
    '''
    # 配置模型
    model = Unet(net_params=net_params).cuda()
    logging.info('Initialize model.')

    # 配置 summarywriter
    shutil.rmtree(snap_path, ignore_errors=True)
    os.makedirs(snap_path)
    writer = SummaryWriter(log_dir=snap_path)

    # 配置 optimizer
    optimizer = optim.SGD(model.parameters(), lr=train_params['learning_rate'], momentum=train_params['momentum'],
                              weight_decay=train_params['weight_decay'])
    logging.info('Reset optimizer.')


    bce_criterion = nn.BCELoss()
    dice_criterion = dice_loss

    loader_iter = iter(loader_train)
    global_step = 0
    best_cur_dice = 0
    lr_ = train_params['learning_rate']

    while global_step < train_params['iterations']:
        try:
            batch = next(loader_iter)
        except:
            loader_iter = iter(loader_train)
            batch = next(loader_iter)

        model.train()
        img, gt = batch['img'].cuda(), batch['gt'].cuda()
        pred = model(img)
        loss_bce = bce_criterion(pred, gt)
        loss_dice = dice_criterion(pred[:, 1, :, :], gt[:, 1, :, :])
        loss = 0.5 * loss_bce + 0.5 * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # 打印训练结果
        if global_step % train_params['print_freq'] == 0:
            logging.info('Iteration %d (lr-%.4f): loss_bce : %.4f; loss_dice: %.4f; loss: %.4f; ' %
                         (global_step, optimizer.param_groups[0]['lr'], loss_bce.item(), loss_dice.item(), loss.item()))

        if global_step % train_params['val_freq'] == 0:
            all_dices = inference(model, loaders_val)
            pre_dices = {}
            for i in loaders_val.keys():
                pre_dices[i] = all_dices[i]
                if i == site:
                    break
            logging.info(f'----------------------> val_dice on %s is %.4f, on previous sites is {pre_dices}' %
                         (site, all_dices[site]))

            cur_dice = all_dices[site]
            pre_dice = np.mean([v for v in pre_dices.values()])

            # 记录summary writer
            writer.add_scalar('train/lr', lr_, global_step)
            writer.add_scalar('loss/loss', loss, global_step)
            writer.add_scalar('loss/loss_bce', loss_bce, global_step)
            writer.add_scalar('loss/loss_dice', loss_dice, global_step)
            writer.add_scalar('val/current_dice', cur_dice, global_step)
            writer.add_scalar('val/pre_dice', pre_dice, global_step)

            # 保存最优模型
            if cur_dice >= best_cur_dice:
                ckpt_path = os.path.join(snap_path, 'best_model.pth')
                torch.save(model.state_dict(), ckpt_path)
                logging.info(
                    '----------------------> cur_dice improved from %.4f to %.4f' % (best_cur_dice, cur_dice))
                best_cur_dice = cur_dice
            else:
                logging.info('----------------------> val_dice did not improved from %.4f' % (best_cur_dice))

            # 定期保存模型
            if global_step % train_params['save_freq'] == 0:
                ckpt_path = os.path.join(snap_path,
                                         'step_%d_curdice_%.4f_predice_%.4f.pth' %
                                         (global_step, cur_dice, pre_dice))
                torch.save(model.state_dict(), ckpt_path)
                logging.info('Iteration %d: save model to %s' % (global_step, ckpt_path))

            if global_step % train_params['lr_decay_freq'] == 0:
                lr_ = train_params['learning_rate'] * 0.95 ** (global_step // train_params['lr_decay_freq'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
    return model

def continual_train_rounds(loaders_train: dict, loaders_val: dict, common_params:dict,
                           net_params:dict, train_params: dict):
    sites = [v for v in loaders_train.keys()]
    for site_index, site in enumerate(sites):
        logging.info(f'============= Train on {site}')

        # 在当前中心训练模型
        single_train_round(site, loader_train=loaders_train[site], loaders_val=loaders_val,
                                   net_params=net_params, train_params=train_params,
                                   snap_path=os.path.join(common_params['exp_root_dir'], site))

def main():
    # 配置文件
    settings = Settings()
    common_params, data_params, net_params, train_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING']

    # 新建任务文件夹
    shutil.rmtree(common_params['exp_root_dir'], ignore_errors=True)
    os.makedirs(common_params['exp_root_dir'])
    logging.basicConfig(filename=os.path.join(common_params['exp_root_dir'], 'logs.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_root_dir'])


    # 配置多中心训练测试数据
    sites = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
    logging.info('============= Multi-Site Training Datasets')
    datasets_train = {site: SeparateDataset(npz_root_dir=os.path.join(data_params['npz_root_dir'], site), mode='train')
                      for site in sites}
    logging.info('============= Multi-Site Testing Datasets')
    datasets_val = {site: SeparateDataset(npz_root_dir=os.path.join(data_params['npz_root_dir'], site), mode='test')
                     for site in sites}
    loaders_train = {site: DataLoader(dataset=dataset, batch_size=train_params['batch_size'], shuffle=True,
                                      num_workers=4, drop_last=True, pin_memory=True)
                     for site, dataset in datasets_train.items()}
    loaders_val = {site: DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                     num_workers=4, drop_last=False, pin_memory=True)
                    for site, dataset in datasets_val.items()}


    # 训练模型
    continual_train_rounds(loaders_train, loaders_val, common_params, net_params, train_params)


if __name__ == '__main__':
    main()
