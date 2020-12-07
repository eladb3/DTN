import torch 
from torch import nn, optim
from torch.nn import functional as F
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

mse_loss = nn.MSELoss(reduction = 'mean')
def dist(x,y):
    return mse_loss(x,y)

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

    xt_rgb =utils.GrayScaleToRGB(x['t'])
    xG = g(f(torch.cat([x['s'], xt_rgb], dim = 0)))
    xcat = torch.cat([xG, xt_rgb]) #[G(x_s), G(x_t), x_t]
    ycat = torch.cat([torch.ones(len(d)) * i for i, d in enumerate([x['s'], x['t'], x['t']])],dim = 0).type(torch.long).to(device)
    xcatD = D(xcat)
    L_D = F.cross_entropy(xcatD, ycat)
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
    
    xcatD = D(torch.cat([xG['s'], xG['t']], dim = 0))
    ycat = torch.ones((xcatD.size(0), )).type(torch.long).to(device)
    L_GANG = F.cross_entropy(xcatD, ycat)
    L_CONST = dist(xf['s'], f(xG['s'][:, [0], :, :])).mean()
#     print(dist(xf['s'], f(xG['s'][:, [0], :, :])).shape)
    L_TID = dist(x['t'], xG['t']).mean()
    
#     L_TV = utils.tvloss(xG['t']) + utils.tvloss(xG['s'])
#     print([L.item() for L in (L_TID , L_GANG , L_CONST)])
#     return 15 * L_TID + L_GANG + 15 * L_CONST
    return 15 * L_TID + L_GANG + 15 * L_CONST


def get_next_batch(x, iters):
    for c in ('s', 't'): x[c] = next(iters[c], None)
    if x['s'] is None or x['t'] is None: return False
    for c in ('s', 't'): x[c] = x[c][0]
    return True

def train(batch_size, Epochs = float("Inf"), save = True, cont = False, plt_ = True , hours = float("Inf")):
    lr = 0.0003
    base = f"./models/trainings/{utils.get_local_time()}"
    os.mkdir(base)
    hours = hours * (60 * 60) # sec in hour
    if cont:
        g = torch.load(f"{cont}/g_net")
        D = torch.load(f"{cont}/D_net")
    else:
        g = Nets.g_net
        D = Nets.D_net
    g, D = g.to(device).train(), D.to(device).train()
    f = Nets.f_net_features.to(device).eval()
#     opt_g = optim.Adam(Nets.g_net.parameters(), lr=  1e-4)
#     opt_D = optim.Adam(Nets.D_net.parameters(), lr=  1e-5)
    opt_g = optim.Adam(Nets.g_net.parameters(), lr=  lr)
    opt_D = optim.Adam(Nets.D_net.parameters(), lr=  lr)
    
    opts = {'g':opt_g, 'D':opt_D}
    loss_fn = {'g':get_g_loss, 'D':get_D_loss}
    dl = {'s':utils.get_svhn(batch_size) , 't':utils.get_mnist(batch_size)}
    dl_test = utils.get_svhn(32, "test")
    dl_test_mnist = utils.get_mnist(32, "test")
    
    for param in Nets.f_net.parameters():
        param.requires_grad = False 
    
    l_loss = {'g':[], 'D':[]}
    l_norm = {'g':[], 'D':[]}
    start_time = time.time()
    e = 0
    while True:
        c = 'g'
        cum_norm, cum_loss, n = {c:[] for c in ('g', 'D')}, {c:[] for c in ('g', 'D')}, {c:0 for c in ('g', 'D')}
        
        def tr_step(c, x):
            loss = loss_fn[c](x, f, g, D)
            opts[c].zero_grad() ; loss.backward() 
            opts[c].step()
            cum_loss[c].append(loss.cpu().item())
            cum_norm[c].append(utils.get_grad_norm(opts[c]))
            n[c] += 1
        
        def myplt(loss, norm):
            g.eval() ; D.eval()
            print(f" >>>> Epoch {e} - Svhn")
            
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            y = g(f(x))
            utils.plt_row_images(y.cpu())
            plt.show()

            print(f" >>>> Epoch {e} - Mnist")
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test_mnist))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            with torch.no_grad():
                y = g(f(x))
            utils.plt_row_images(y.cpu())
            plt.show()
            
            print(f" >>>> Epoch {e} - losses")
            plt.figure(figsize = (15, 10))
            utils.plt_losses_norm(loss, norm, start_from = 2)
            plt.show()
            g.train() ; D.train()

        
        
        x = {}
        iters = {k:iter(dl[k]) for k in dl}
        t = 0
        while get_next_batch(x, iters):
            for k in x: x[k] = utils.GrayScaleToRGB(x[k]).to(device)
            tr_step('D', x)
            for i in range(10):
                tr_step('g', x)

            if t % 50 == 0 and t > 1: myplt(cum_loss, cum_norm)
            t += 1
            
        for c in ('g', 'D'):
            l_loss[c].append(sum(cum_loss[c]) / n[c])
            l_norm[c].append(sum(cum_norm[c]) / n[c])
        print(f">>> Epoch: {e}, loss G: {l_loss['g'][-1]}, loss D: {l_loss['D'][-1]}")
        
        if plt_:
            g.eval() ; D.eval()
            print(f" >>>> Epoch {e} - Svhn")
            
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            y = g(f(x))
            utils.plt_row_images(y.cpu())
            plt.show()

            print(f" >>>> Epoch {e} - Mnist")
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test_mnist))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            with torch.no_grad():
                y = g(f(x))
            utils.plt_row_images(y.cpu())
            plt.show()
            
            print(f" >>>> Epoch {e} - losses")
            plt.figure(figsize = (15, 10))
            utils.plt_losses_norm(l_loss, l_norm, start_from = 2)
            plt.show()
            g.train() ; D.train()
        if save and e > 0 and e % 10 == 0:
            os.mkdir(f"{base}/{e}")
            for name, model in [("g_net", g), ("D_net", D)]:
                torch.save(model.cpu(), f"{base}/{e}/{name}_epoch_{e}")
                model.to(device)
            torch.save({'loss':l_loss, 'norm':l_norm}, f"{base}/{e}/hist")
            print(f"CP -- models saved")
        
        e += 1
        if e > Epochs: break
        if (time.time() - start_time) > hours: break
        
    
    if save:
        for name, model in [("g_net", g), ("D_net", D)]:
            torch.save(model.cpu(), f"{base}/{name}")
    
    utils.rmdir_if_empty(base)
        