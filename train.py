from util.data import *
from model.Unet import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DIR = "D:\\dataset\\BraTS2020\\MICCAI_BraTS2020_TrainingData"

net_params = {'num_filters':32, 'num_channels':1, 'num_classes':1}
model = Unet(net_params)

data = BraTS_SLICE(DATA_DIR)
loader = DataLoader(data, batch_size=5, shuffle=True)
for image, seg in loader:
    print(image.shape)
    pred = model(image)
    plt.imshow(pred, cmap=plt.cm.gray)
    print(pred.shape)



