import torch
from torch import nn, optim
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from copy import deepcopy
import random
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(device, flush = True)

## plt

def plt_tensor(x):
    x = (x+1)*0.5
    plt.imshow(TF.to_pil_image(x))
    
def get_local_time():
    lt = time.localtime()
    return f"{lt.tm_mday}_{lt.tm_mon}_{lt.tm_year}_{lt.tm_hour}_{lt.tm_min}_{lt.tm_sec}"

def rmdir_if_empty(path):
    if not os.path.isdir(path): return
    if [f for f in os.listdir(path) if not f.startswith('.')] == []:
        os.rmdir(path)

def get_grad_norm(opt):
    return sum([w.grad.norm().cpu().item() for w in opt.param_groups[0]['params']])

def plt_losses_norm(l_loss, l_norm, start_from = 0):
    
    cs = list(l_loss.keys())
    i = 1
    start = start_from
    if len(l_loss[list(l_loss.keys())[0]]) < start_from + 1: start = 0
    for c in cs:
        plt.subplot(len(cs), 2, i)
        plt.title(f"{c} Loss")
        plt.plot(l_loss[c][start:])
        i += 1
        plt.subplot(len(cs),2,i)
        plt.title(f"{c} Norm")
        plt.plot(l_norm[c][start:])
        i += 1

def plt_row_images(y):
    n = y.size(0)
    for i in range(n):
        plt.subplot(1,n,i+1)
        img = y[i, :, :, :].cpu()
        plt_tensor(img)

    
## data

def get_svhn(batch_size, split = "train"):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    d = torchvision.datasets.SVHN(root="./data/SVHN", split = split, download = True, transform = t)
    return DataLoader(d, batch_size = batch_size, shuffle = True)

def get_mnist(batch_size, split = "train"):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Lambda(lambda x: x * 2 - 1),
#         transforms.Normalize((0.5,), (0.5,))
    ])
    train = split == "train"
    d = torchvision.datasets.MNIST(root="./data/MNIST", train = train, download = True, transform = t)
    return DataLoader(d, batch_size = batch_size, shuffle = True)



## Loss

def tvloss(imgs):
    return 0

## Layers

class nn_Reshape(nn.Module):
    def __init__(self, C, H ,W):
        super(nn_Reshape, self).__init__()
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        x = x.view((x.size(0), self.C, self.H ,self.W))
        return x

class nn_Print(nn.Module):
    def __init__(self):
        super(nn_Print, self).__init__()
    def forward(self, x):
        print(1)
        return x

class nn_GrayScaleToRGB(nn.Module):
    def __init__(self):
        super(nn_GrayScaleToRGB, self).__init__()
    
    def forward(self, x):
        if x.size(1) == 1:
            x = x.expand(x.size(0), 3, x.size(2), x.size(3))
        return x
    
class nn_Tanh_to_Img(nn.Module):
    def __init__(self):
        super(nn_Tanh_to_Img, self).__init__()
    
    def forward(self, x):
        x = (x+1)/2
        return x
    
    
def GrayScaleToRGB(x):
    if x.size(1) == 1:
        x = x.expand(x.size(0), 3, x.size(2), x.size(3))
    return x

class nn_Cut(nn.Module):
    def __init__(self, C = None, H = None, W = None):
        super(nn_Cut, self).__init__()
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        if self.C:
            x = x[:, :self.C, :, :]
        if self.H:
            x = x[:,:, :self.H, :]
        if self.W:
            x = x[:, :, :, :self.W]
        return x


### Train
def print_during_train(epoch, loss, acc):
    s = f'>>> Epoch {epoch}, '
    s += f"train: loss {loss['train'][-1]:.4f} acc {acc['train'][-1]:.4f}, "
    s += f"test: loss {loss['test'][-1]:.4f} acc {acc['test'][-1]:.4f}, "
    print(s, flush = True)
    
def plt_loss_acc(loss, acc, path):
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title("Accuracy")
    plt.plot(acc['train'], label = 'train')
    plt.plot(acc['test'], label = 'test')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title("Loss")
    plt.plot(loss['train'], label = 'train')
    plt.plot(loss['test'], label = 'test')
    plt.legend()
    plt.savefig(f"{path}/train_plot.jpg")


def train_Net(Net, Epochs, dl, dl_test, path = None, add_softmax = False, print_each = 100, early_stop = 10, with_tqdm = False):
    if path is not None:
        if not os.path.isdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)
    Net.to(device)
    opt = optim.Adam(Net.parameters(), lr=  1e-4)
    loss_fn = nn.CrossEntropyLoss()
    E = Epochs
    l_loss, l_acc = {'train':[], 'test':[]}, {'train':[], 'test':[]}
    best_model, best_acc = None ,0
    n_train, n_test = len(dl.dataset), len(dl_test.dataset)
    early_stop_state = 0
    for epoch in range(E):
        epoch_loss, epoch_acc = 0, 0
        # train epoch
        rng = tqdm(dl) if with_tqdm else dl
        for x, y in rng:
            x, y = x.to(device), y.view(-1).type(torch.int64).to(device)
            y_pred = Net(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad() ; loss.backward() ; opt.step()
            with torch.no_grad():
                c_acc = (y_pred.cpu().argmax(axis=1) == y.cpu()).type(torch.float32).sum()
                epoch_acc += c_acc
            epoch_loss += loss * len(x) 
        l_loss['train'].append(epoch_loss / n_train)
        l_acc['train'].append(epoch_acc / n_train)
        epoch_loss, epoch_acc = 0, 0
        # test
        rng = tqdm(dl_test) if with_tqdm else dl_test
        for x,y in rng:
            x, y = x.to(device), y.view(-1).type(torch.int64).to(device)
            with torch.no_grad():
                y_pred = Net(x)
                loss = loss_fn(y_pred, y)
                epoch_acc += (y_pred.cpu().argmax(axis=1) == y.cpu()).type(torch.float32).sum()
            epoch_loss += loss * len(x)
        # update best model
        l_loss['test'].append(epoch_loss / n_test)
        l_acc['test'].append(epoch_acc / n_test)
        if l_acc['test'][-1] > best_acc:
            best_acc = l_acc['test'][-1]
            best_model_train_acc = l_acc['train'][-1]
            best_model = deepcopy(Net)
            early_stop_state = 0
        else: 
            early_stop_state += 1
        if early_stop_state > early_stop:
            print(f"Test accuracy has not improved in {early_stop_state} epochs, stop train", flush = True)
            break
        if epoch % print_each == 0: 
            print_during_train(epoch,l_loss, l_acc)
    print(f"Best model test accuracy: {best_acc}, train accuracy: {best_model_train_acc}", flush = True)
    if path is not None:
        if add_softmax: best_model = nn.Sequential(best_model, nn.Softmax(dim = 1))
        torch.save(best_model.cpu(), f"{path}/model")
        print(f">>> model saved to {path}", flush = True)
    plt_loss_acc(l_loss, l_acc, path = path)
