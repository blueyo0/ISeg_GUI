from util.data import *
from model.Unet import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 修复OMP报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DATA_DIR = "D:/dataset/BraTS2020/MICCAI_BraTS2020_TrainingData"
MODEL_PATH = "D:/code/Model_File"

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

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = diceCoeff
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        torch.save(model, os.path.join(MODEL_PATH, "unet2d_epoch{:03d}.pth".format(epoch)))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# net_params = {'num_filters':32, 'num_channels':1, 'num_classes':1}
# model = Unet(net_params)

data = BraTS_SLICE(DATA_DIR)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True) 


# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_ch5(net=model, train_iter=train_loader, test_iter=test_loader, 
#           batch_size=5, optimizer=optimizer, device='cuda', num_epochs=10)

model = torch.load(os.path.join(MODEL_PATH, "unet2d_epoch{:03d}.pth".format(9)))
model.eval()

for img, seg in test_loader:
    with torch.no_grad():
        pred = model(img.to("cuda"))
    plt.imshow(pred.cpu().detach().numpy()[0,0,:,:])
    plt.show()


