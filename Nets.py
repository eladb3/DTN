import torch 
from torch import nn
from torch import functional as F
import numpy as np
import scipy
from torchvision import models, transforms
import torchvision
import scipy
import utils
from collections import OrderedDict

## g net
g_in_dim = (128)
g_out_dim = (3, 32, 32)
g_net = nn.Sequential(
    utils.nn_Reshape(32, 2, 2),
    nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding = 0),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2, padding = 0),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, kernel_size = 2, stride = 2, padding = 0),
    nn.BatchNorm2d(4),
    nn.ReLU(),
    nn.ConvTranspose2d(4, 1, kernel_size = 2, stride = 2, padding = 0),
    nn.Tanh(),
    utils.nn_GrayScaleToRGB()
)
f_net = torch.load("./models/f_net/model")
f_net_features = f_net._modules['0']


G_net = nn.Sequential(OrderedDict([
    ('f', f_net_features),
    ('g', g_net)
]))


# Discriminator

D_in_dim = (3, 32, 32)
D_out_dim = (3)
D_net = nn.Sequential(
    utils.nn_GrayScaleToRGB(),
    nn.Conv2d(3, 4, kernel_size = 3, stride =2, padding = 1),
    nn.BatchNorm2d(4),
    nn.ReLU(),
    nn.Conv2d(4, 4, kernel_size = 3, stride = 2, padding = 1),
    nn.BatchNorm2d(4),
    nn.ReLU(),
    nn.Conv2d(4, 4, kernel_size = 3, stride = 2, padding = 1),
    nn.BatchNorm2d(4),
    nn.ReLU(),
    nn.Conv2d(4, 3, kernel_size = 4, stride = 1, padding = 0),
    nn.Flatten(),
    nn.Softmax(dim = 1)
)

## Older
# First version
# D_net = nn.Sequential(
#     utils.nn_GrayScaleToRGB(),
#     nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
#     nn.BatchNorm2d(16),
#     nn.ReLU(),
#     nn.Conv2d(16, 4, kernel_size = 3, stride = 1, padding = 1),
#     nn.BatchNorm2d(4),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(4096, 16),
#     nn.ReLU(),
#     nn.Linear(16, 3),
#     nn.Softmax(dim = 1)
# )

# D_net = nn.Sequential(
#     utils.nn_GrayScaleToRGB(),
#     nn.Conv2d(3, 4, kernel_size = 3, stride =2, padding = 1),
#     nn.BatchNorm2d(4),
#     nn.ReLU(),
#     nn.Conv2d(4, 4, kernel_size = 3, stride = 2, padding = 1),
#     nn.BatchNorm2d(4),
#     nn.ReLU(),
#     nn.Conv2d(4, 4, kernel_size = 3, stride = 1, padding = 1),
#     nn.BatchNorm2d(4),
#     nn.ReLU(),
#     nn.Conv2d(4, 1, kernel_size = 3, stride = 1, padding = 1),
#     nn.BatchNorm2d(1),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(1*  8 * 8, 16),
#     nn.ReLU(),
#     nn.Linear(16, 3),
#     nn.Softmax(dim = 1)
# )
