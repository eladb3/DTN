import torch 
from torch import nn
from torch import functional as F
import numpy as np
import scipy
from torchvision import models, transforms
import torchvision
import scipy
from PIL import Image
from torchsummary import summary
from torch.utils.data import DataLoader
import utils
device = torch.device('cuda')


class nn_GrayScaleToRGB(nn.Module):
    def __init__(self):
        super(nn_GrayScaleToRGB, self).__init__()
    
    def forward(self, x):
        if x.size(1) == 1:
            x = x.expand(x.size(0), 3, x.size(2), x.size(3))
        return x

class nn_Print(nn.Module):
    def __init__(self):
        super(nn_Print, self).__init__()
    
    def forward(self, x):
        print(f"nn_Print: x shape - {x.shape}")
        return x

    
f_in_dim = (3, 32,32)
f_out_dim = (128)

f_net_features = nn.Sequential(
    utils.nn_GrayScaleToRGB(),
    nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
    nn.ReLU(),
    nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 4, stride = 1, padding = 0),
    nn.ReLU(),
    nn.Flatten()
)

f_net_classifier = nn.Sequential(
    nn.Linear(128, 10)
)

f_net = nn.Sequential(
    f_net_features,
    f_net_classifier
)




if __name__ == "__main__":
    batch_size = 256
    Epochs = 1000
    early_stop = 10
    
    t = transforms.Compose([
        transforms.ToTensor()
    ])
    d = torchvision.datasets.SVHN(root="./data/SVHN", split = "extra", download = True, transform = t)
    validation_size = int(len(d) * 0.1)
    train_data, validation_data = torch.utils.data.random_split(d,(len(d) - validation_size, validation_size))
    dl_train, dl_test = DataLoader(train_data, batch_size = batch_size), DataLoader(validation_data, batch_size = batch_size)
    
    ## training
    utils.train_Net(f_net, Epochs, dl_train, dl_test, path = "./models/f_net", add_softmax = False, print_each = 1, early_stop = 10)
        