from util.data import *
from model.Unet import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 修复OMP报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import platform
sysstr = platform.system()
if(sysstr=="Windows"):
    DATA_DIR = "D:/dataset/BraTS2020/MICCAI_BraTS2020_TrainingData"
    MODEL_PATH = "D:/code/Model_File"
elif(sysstr=="Linux"):
    DATA_DIR = "/opt/data/private/why/BraTS2020/MICCAI_BraTS2020_TrainingData"
    MODEL_PATH = "/opt/data/private/why/model"

def diceCoeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    # loss = nn.MSELoss()
    loss = diceCoeff
    for epoch in range(num_epochs):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat[:,0,:,:].unsqueeze(1)
            l = 1-loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            print((u'\r'+str(batch_count)), end=' ')
        test_acc = evaluate_accuracy(test_iter, net)
        torch.save(model, os.path.join(MODEL_PATH, "unet2d_epoch{:03d}.pth".format(epoch)))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


MODE = 'train'
# MODE = 'test'

DEVICE = 'cuda'
# DEVICE = 'cpu'

data = BraTS_SLICE(DATA_DIR)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True) 

# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
if(MODE=='train'):
    num_classes = 3
    net_params = {'num_filters':32, 'num_channels':1, 'num_classes':num_classes}
    model = Unet(net_params).to(DEVICE)
    # for img, seg in train_loader:
    #     pred = model(img.to(DEVICE))
    #     for ix in range(5):
    #         for iy in range(num_classes):
    #             plt.subplot(num_classes+1,5,ix+iy*5+1)
    #             plt.imshow(pred.cpu().detach().numpy()[ix,iy,:,:])
    #     for ix in range(5):
    #         plt.subplot(num_classes+1,5,num_classes*5+1+ix)
    #         plt.imshow(seg.cpu().detach().numpy()[ix,0,:,:])
    #     plt.show()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(net=model, train_iter=train_loader, test_iter=test_loader, 
                batch_size=5, optimizer=optimizer, device=DEVICE, num_epochs=10)

elif(MODE=='test'):
    model = torch.load(os.path.join(MODEL_PATH, "unet2d_epoch{:03d}.pth".format(9)))
    net_params = {'num_filters':32, 'num_channels':1, 'num_classes':3}
    model = Unet(net_params).to(DEVICE)
    model.eval()

    for img, seg in test_loader:
        with torch.no_grad():
            pred = model(img.to(DEVICE)).cpu().detach().numpy()

        for i in range(5):
            max_dice = 0.0
            max_threshold = 0.1
            for threshold in np.arange(0.1, 0.7, 0.01):
                pred_copy = pred.copy()
                for x in np.nditer(pred_copy, op_flags=['readwrite']):
                    x[...]= 1.0 if(x<threshold) else 0.0
                dice = diceCoeff(torch.from_numpy(pred_copy[0,0,:,:]), seg[0,0,:,:])
                print(threshold, dice, end='\r')
                if(dice > max_dice):
                    max_threshold = threshold
                

            for x in np.nditer(pred, op_flags=['readwrite']):
                x[...]= 1.0 if(x>max_threshold) else 0.0

            plt.subplot(1,3,1)
            plt.imshow(pred[i,0,:,:])
            plt.subplot(1,3,2)
            plt.imshow(img.cpu().detach().numpy()[i,0,:,:])
            plt.subplot(1,3,3)
            plt.imshow(seg.cpu().detach().numpy()[i,0,:,:])
            plt.show()


