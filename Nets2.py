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


def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    except:
        pass
    


## g net
g_in_dim = (128)
g_out_dim = (3, 32, 32)

# g_net = nn.Sequential(
#     utils.nn_Reshape(128, 1, 1),
#     nn.ConvTranspose2d(128, 512, kernel_size = 4, stride = 2, padding = 0),
#     nn.BatchNorm2d(512),
#     nn.ReLU(),
#     nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1),
#     nn.BatchNorm2d(256),
#     nn.ReLU(),
#     nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
#     nn.BatchNorm2d(128),
#     nn.ReLU(),
#     nn.ConvTranspose2d(128, 1, kernel_size = 4, stride = 2, padding = 1),
#     nn.Tanh(),
#     utils.nn_GrayScaleToRGB()
# )


# g_net = nn.Sequential(
#     utils.nn_Reshape(32, 2, 2),
#     nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding = 0),
#     nn.BatchNorm2d(16),
#     nn.ReLU(),
#     nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2, padding = 0),
#     nn.BatchNorm2d(8),
#     nn.ReLU(),
#     nn.ConvTranspose2d(8, 4, kernel_size = 2, stride = 2, padding = 0),
#     nn.BatchNorm2d(4),
#     nn.ReLU(),
#     nn.ConvTranspose2d(4, 1, kernel_size = 2, stride = 2, padding = 0),
#     nn.Tanh(),
#     utils.nn_Tanh_to_Img(),
#     utils.nn_GrayScaleToRGB()
# )


n_noise, ngf, nc = 128, 64, 1
g_net = nn.Sequential(
    utils.nn_Reshape(-1, 1,1), #(, 128, 1, 1)
    nn.ConvTranspose2d(n_noise, ngf * 4, kernel_size = 4, stride = 1, padding = 0), #(bs, ngf*4, 4, 4)
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(),
    nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size = 4, stride = 2, padding = 1),#(bs, ngf*2, 8, 8)
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(),
    nn.ConvTranspose2d(ngf * 2, ngf, kernel_size = 4, stride = 2, padding = 1),#(bs, ngf, 16, 16)
    nn.BatchNorm2d(ngf),
    nn.ReLU(),
    nn.ConvTranspose2d(ngf, nc, kernel_size = 4, stride = 2, padding = 1),#(bs, nc = 1, 32, 32)
    nn.Tanh(),
    utils.nn_GrayScaleToRGB()
)

g_net.apply(init_weights)



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
    nn.Conv2d(3, 128, kernel_size = 3, stride =2, padding = 1),
    nn.LeakyReLU(0.2),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),
    nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2),
    nn.Conv2d(512, 1, kernel_size = 4, stride = 2, padding = 0),
    nn.Flatten(),
    nn.Sigmoid(),
)


D_net.apply(init_weights)

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
