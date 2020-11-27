import torch 
from torch import nn, optim
from torch import functional as F
import numpy as np
import scipy
from torchvision import models, transforms
import torchvision
import scipy
import utils
from collections import OrderedDict
import Nets
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import os
device = torch.device('cuda')

def dist(x, y):
    return ((x-y)**2).sum()
    
def get_loss(x, f, g, D):
    """
    x: dict with keys : 's':(N, 3, 32, 32), 't':(N,3,32,32)
    """
    xf, xG = {}
    for n in ["s", "t"]:
        xf[n]  = f(x[n])
        xG[n] = g(xf[n])

    xD = {}
    xD['G_s'] = D(xG['s'])
    xD['G_t'] = D(xG['t'])
    xD['t'] = D(x['t'])
    
    L_D = -sum(xD[n][:, i].log().mean() for i, n in enumerate(['G_x', 'G_t', 't']))
    
    L_GANG = -xD['G_t'][:, 2].log().mean() -xD['G_s'][:, 2].log().mean() 
    
    L_CONST = dist(xf['s'], f(xG['s'])).sum()
    
    L_TID = dist(x['t'], xG['t']).sum()
    
    L_TV = utils.tvloss(xG['t']) + utils.tvloss(xG['s'])
    
    return L_D + L_GANG + L_CONST + L_TID + L_TV


def get_D_loss(x, f, g, D):
    """
    x: dict with keys : 's':(N, 3, 32, 32), 't':(N,3,32,32)
    """
    xf, xG = {}, {}
    for n in ["s", "t"]:
        xf[n]  = f(x[n])
        xG[n] = g(xf[n])

    xD = {}
    xD['G_s'] = D(xG['s'])
    xD['G_t'] = D(xG['t'])
    xD['t'] = D(x['t'])

    L_D = -sum(xD[n][:, i].log().mean() for i, n in enumerate(['G_s', 'G_t', 't']))
        
    return L_D

def get_g_loss(x, f, g, D):
    """
    x: dict with keys : 's':(N, 3, 32, 32), 't':(N,3,32,32)
    """
    xf, xG = {}, {}
    for n in ["s", "t"]:
        xf[n]  = f(x[n])
        xG[n] = g(xf[n])

    xD = {}
    xD['G_s'] = D(xG['s'])
    xD['G_t'] = D(xG['t'])
    
    L_GANG = -xD['G_t'][:, 2].log().mean() -xD['G_s'][:, 2].log().mean() 
    
    L_CONST = dist(xf['s'], f(xG['s'][:, [0], :, :])).sum()
    
    L_TID = dist(x['t'], xG['t']).sum()
    
    L_TV = utils.tvloss(xG['t']) + utils.tvloss(xG['s'])
    
    return 15 * L_TID + L_GANG + 15 * L_CONST



def get_next_batch(x, iters):
    for c in ('s', 't'): x[c] = next(iters[c], None)
    if x['s'] is None or x['t'] is None: return False
    for c in ('s', 't'): x[c] = x[c][0]
    return True

def train(batch_size, Epochs = float("Inf"), save = True, cont = False, plt_ = True , hours = float("Inf")):
    base = f"./models/trainings/{utils.get_local_time()}"
    os.mkdir(base)
    hours = hours * (60 * 60) # sec in hour
    if cont:
        g = torch.load(f"{base}/g_net")
        D = torch.load(f"{base}/D_net")
    else:
        g = Nets.g_net
        D = Nets.D_net
    g, D = g.to(device), D.to(device)
    f = Nets.f_net_features.to(device)
    opt_g = optim.Adam(Nets.g_net.parameters(), lr=  1e-3)
    opt_D = optim.Adam(Nets.D_net.parameters(), lr=  5e-5)
    
    opts = {'g':opt_g, 'D':opt_D}
    loss_fn = {'g':get_g_loss, 'D':get_D_loss}
    dl = {'s':utils.get_svhn(batch_size) , 't':utils.get_mnist(batch_size)}
    dl_test = utils.get_svhn(10, "test")
    dl_test_mnist = utils.get_mnist(10, "test")
    
    for param in Nets.f_net.parameters():
        param.requires_grad = False 
    
    l_loss = {'g':[], 'D':[]}
    l_norm = {'g':[], 'D':[]}
    start_time = time.time()
    e = 0
    while True:
        c = 'g'
        cum_loss, n = {c:0 for c in ('g', 'D')}, {c:0 for c in ('g', 'D')}
        x = {}
        iters = {k:iter(dl[k]) for k in dl}
        while get_next_batch(x, iters):
            for k in x: x[k] = x[k].to(device)
            loss = loss_fn[c](x, f, g, D)
            opts[c].zero_grad() ; loss.backward() ; opts[c].step()
            l_norm[c].append(utils.get_grad_norm(opts[c]))
            cum_loss[c] += loss.cpu().item()
            n[c] += 1
            c = 'g' if sum(n[k] for k in n) % 10 else 'D' # 9 times g, 1 time D
        for c in ('g', 'D'): l_loss[c].append(cum_loss[c] / n[c])
        print(f">>> Epoch: {e}, loss G: {l_loss['g'][-1]}, loss D: {l_loss['D'][-1]}")
        
        if plt_:
            print(f" >>>> Epoch {e}")
            plt.figure(figsize = (15, 10))
            x ,y  = next(iter(dl_test))
            x = x.to(device)
            with torch.no_grad():
                y = g(f(x))

            for i in range(4):
                plt.subplot(1,4,i+1)
                img = y[i, :, :, :].cpu()
                utils.plt_tensor(img)
            plt.show()

            print(f" >>>> Epoch {e} - Mnist")
            plt.figure(figsize = (15, 10))
            x ,y  = next(iter(dl_test_mnist))
            x = x.to(device)
            with torch.no_grad():
                y = g(f(x))

            for i in range(4):
                plt.subplot(1,4,i+1)
                img = y[i, :, :, :].cpu()
                utils.plt_tensor(img)
            plt.show()
            
            print(f" >>>> Epoch {e} - losses")
            plt.figure(figsize = (15, 10))
            utils.plt_losses_norm(l_loss, l_norm)
            plt.show()

        if save and e > 0 and e % 1 == 0:
            for name, model in [("g_net", g), ("D_net", D)]:
                torch.save(model.cpu(), f"{base}/{name}_epoch_{e}")
                model.to(device)
            print(f"CP -- models saved")
        
        e += 1
        if e > Epochs: break
        if (time.time() - start_time) > hours: break
        
    
    if save:
        for name, model in [("g_net", g), ("D_net", D)]:
            torch.save(model.cpu(), f"{base}/{name}")
    
    utils.rmdir_if_empty(base)
        