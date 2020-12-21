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
import Nets2 as Nets
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import os
import pickle
device = torch.device('cuda')

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss(reduction = 'mean')
def dist(x,y):
    return mse_loss(x,y)

def get_rand_noise(n):
    r = torch.randn((n,)).abs() * 0.08
    r[r>0.3] = 0.3
    return r

def get_loss(n, x, f, g, D, weight):
    """
    x: dict with keys : 's':(N, 3, 32, 32), 't':(N,3,32,32)
    """
    
    if n == 'L_GANG':
        loss = 0
        for k in x:
            xc = g(f(x[k]))
            xD = D(xc)
            y = torch.ones((xD.size(0), )).to(device)
            loss += bce_loss(xD.view(-1), y) * weight

#         xcatD = D(torch.cat([g(f(x['s'])), g(f(x['t']))], dim = 0))
#         ycat = torch.ones((xcatD.size(0), )).type(torch.float).to(device)
#         loss = bce_loss(xcatD.view(-1), ycat) * weight

    elif n == 'L_TID':
        n1 = x['t'] #torch.zeros_like(x['t']).type(torch.float32).to(device)
        n2 = g(f(x['t']))
        loss = dist(n1, n2) * weight
        loss = loss
        
    
    elif n == 'L_CONST':
        loss = dist(f(x['s']), f(g(f(x['s']))[:, [0], :, :])) * weight

    elif n == 'L_D':
        xt_rgb =utils.GrayScaleToRGB(x['t'])
        p = {}
        loss = 0
        for name, cx in (('trans', x['s']), ('real_trans', xt_rgb)):
            xG = g(f(cx))
            y = torch.zeros(cx.size(0)).type(torch.float).to(device) + get_rand_noise(cx.size(0)).to(device)
            xD = D(xG)
            loss += bce_loss(xD.view(-1), y) * weight
            p[name] = xD.detach().mean().cpu()
        xD = D(xt_rgb)
        y = torch.ones(xt_rgb.size(0)).type(torch.float).to(device) - get_rand_noise(cx.size(0)).to(device)
        loss += bce_loss(xD.view(-1), y) * weight
        p['real'] = xD.detach().mean().cpu()

#         xG = g(f(torch.cat([x['s'], xt_rgb], dim = 0)))
#         xcat = torch.cat([xG, xt_rgb]) #[G(x_s), G(x_t), x_t]
#         ycat = torch.cat([torch.ones(len(d)) * i for i, d in [(0, x['s']), (0, x['t']), (1,x['t'])]],dim = 0).type(torch.float).to(device)
#         xcatD = D(xcat)
#         loss = bce_loss(xcatD.view(-1), ycat) * weight
#         probs = xcatD.detach()
#         p['trans'] = probs[:len(x['s'])].mean()
#         p['real_trans'] = probs[len(x['s']) : len(x['s']) + len(x['t'])].mean()
#         p['real'] = probs[len(x['s']) + len(x['t']):].mean()        
        loss = (loss, p)

    else:
        raise Expection("BAD LOSS NAME")
    
    return loss


def get_next_batch(x, iters):
    for c in ('s', 't'): x[c] = next(iters[c], None)
    if x['s'] is None or x['t'] is None: return False
    for c in ('s', 't'): x[c] = x[c][0]
    x['t'] = utils.GrayScaleToRGB(x['t'])
    return True

def train(batch_size, Epochs = float("Inf"), 
          L_times = {},
          weights = {"L_TID":15, "L_CONST":15},
          save = True, cont = False, plt_ = True , hours = float("Inf"), lr = 0.0003, show = False):
    base = f"./models/trainings/{utils.get_local_time()}"
    params = locals()
    os.mkdir(base)
    with open(f"{base}/params.txt", 'wt') as f: f.write(str(params))
    with open(f"{base}/params_dict", 'wb') as f: pickle.dump(params, f)
    hours = hours * (60 * 60) # sec in hour
    
    for k in L_times.keys():
        if isinstance(k, int):
            L_times[k] = lambda x: L_times[k]
            
    
    if cont:
        g = torch.load(f"{cont}/g_net")
        D = torch.load(f"{cont}/D_net")
    else:
        g = Nets.g_net
        D = Nets.D_net
    g, D = g.to(device), D.to(device)
    f = Nets.f_net_features.to(device)
    opt_g = optim.Adam(g.parameters(), lr=  lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=  lr, betas=(0.5, 0.999))
    
    opts = {'g':opt_g, 'D':opt_D}
    dl = {'s':utils.get_svhn(batch_size) , 't':utils.get_mnist(batch_size)}
    dl_test = utils.get_svhn(32, "test")
    dl_test_mnist = utils.get_mnist(32, "test")
    
    names = ['L_GANG', 'L_CONST', 'L_D', 'L_TID']
    prob_types = ['trans', 'real_trans', 'real']
    cum_norm, cum_loss, n = {c:0 for c in names}, {c:[] for c in names}, {c:0 for c in names}
    cum_prob = {typ:[] for typ in prob_types}
    
    def myplt(path = None):
            g.eval() ; D.eval()
            print(f" >>>> Epoch {e} - Svhn")
            
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            if path: plt.savefig(f"{path}/svhn.jpg")
            if show: plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            y = g(f(x))
            utils.plt_row_images(y.cpu())
            if path: plt.savefig(f"{path}/svhn_G.jpg")
            if show: plt.show()

            print(f" >>>> Epoch {e} - Mnist")
            plt.figure(figsize = (15, 10))
            x ,_  = next(iter(dl_test_mnist))
            x = x[:4, :, :, :]
            utils.plt_row_images(x.cpu())
            if path: plt.savefig(f"{path}/mnist.jpg")
            if show: plt.show()
            plt.figure(figsize = (15, 10))
            x = x.to(device)
            with torch.no_grad():
                y = g(f(x))
            utils.plt_row_images(y.cpu())
            if path: plt.savefig(f"{path}/mnist_G.jpg")
            if show: plt.show()
            
            plt.figure(figsize = (15, 10))
            for i, name in enumerate(names):
                plt.subplot(2,2,i+1)
                plt.title(name)
                cp = cum_loss[name]
                if len(cp) > 200: cp = cp[50:]
                plt.plot(cp)
            if path: plt.savefig(f"{path}/loss.jpg")
            if show: plt.show()
            
            plt.figure(figsize = (15, 10))
            for typ in prob_types:
                cp = cum_prob[typ]
                plt.plot(cp, label = typ)
            plt.legend()
            plt.title("Disc Probs")
            if path: plt.savefig(f"{path}/probs.jpg")
            if show: plt.show()
            g.train() ; D.train()

    
    
    
    def tr_step(name, x, times = 1):
        c = 'D' if n in ['L_D'] else 'g'
        if times == 0:
            cum_loss[name].append(cum_loss[name][-1] if len(cum_loss[name]) else 0)
            return
        closs = 0
        cp = {typ:0 for typ in prob_types}
        for i in range(times):
            loss = get_loss(name, x, f, g, D, weight = weights.get(name, 1))
            
            if name == "L_D":
                loss, p = loss
                for typ in p:
                    cp[typ] += p[typ].cpu().item()
    
            closs += loss
            opts[c].zero_grad() ; loss.backward() ; opts[c].step()
        cum_loss[name].append(closs.cpu().item() / times)
        if name == "L_D":
            for typ in prob_types:
                cum_prob[typ].append(cp[typ] / times)
        n[name] += times
    
    start_time = time.time()
    e = 0
    while True:
    
        x = {}
        iters = {k:iter(dl[k]) for k in dl}
        i = 0
        while get_next_batch(x, iters):
            for k in x: x[k] = x[k].to(device)
            
            tr_step('L_D', x, times = L_times.get("L_D", lambda x: 1)(i))
            tr_step('L_GANG', x, times = L_times.get("L_GANG", lambda x: 1)(i))
            tr_step('L_CONST', x, times = L_times.get("L_CONST", lambda x: 0 if x % 10 else 1)(i))
            tr_step('L_TID', x, times = L_times.get("L_TID", lambda x: 1)(i))

            i += 1
            if i % 500 == 0 and i > 1:
                myplt()


        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH {e} END")
        if save and e > 0 and e % 5 == 0:
            os.mkdir(f"{base}/{e}")
            myplt(path = f"{base}/{e}")
            for name, model in [("g_net", g), ("D_net", D)]:
                torch.save(model.cpu(), f"{base}/{e}/{name}")
                model.to(device)
            print(f"CP -- models saved to {base}/{e}/{name}")
        
        e += 1
        if e > Epochs: break
        if (time.time() - start_time) > hours: break
        
    if save:
        for name, model in [("g_net", g), ("D_net", D)]:
            torch.save(model.cpu(), f"{base}/{name}")
    
    utils.rmdir_if_empty(base)
        