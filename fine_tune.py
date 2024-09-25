import os.path
import random

import torch
import numpy as np
from torch import nn
from model import UrbanZ
from function import get_ft_dataloader,get_ft2_dataloader

test_cuda = True if torch.cuda.is_available() else False

torch.manual_seed(2024)

#model=cf_net(num_res=1,in_c=1,out_c=1,num_hid=64)
model=UrbanZ()
if test_cuda:
    model=model.cuda()
#model.load_state_dict(torch.load(os.path.join('paramter','final.pt')))
#model.cf_net.load_state_dict(torch.load(os.path.join('paramter','cf_net.pt')))
#model.st_net.load_state_dict(torch.load(os.path.join('paramter','st_net.pt')))
#model.lt_net.load_state_dict(torch.load(os.path.join('paramter','lt_net.pt')))

loss=nn.MSELoss()
if test_cuda:
    loss=loss.cuda()

train_dataloader=get_ft_dataloader(path='data',batch_size=16, mode='train',shuffle=False)
test_dataloader=get_ft_dataloader(path='data',batch_size=16, mode='test',shuffle=False)

lr=1e-3
trainer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))

epoch=200

hf=30

mn_loss=1e9

for k in range(epoch):
    for i,(X,Y,ext) in enumerate(train_dataloader):

        if X.shape[0]!=16:
            continue
        # X=X.squeeze(0)
        # X=X.permute(1,0,2,3)
        # Y=Y.squeeze(0)
        # Y=Y.permute(1,0,2,3)
        # ext=ext.squeeze(0)
        # ext=ext.permute(1,0,2)
        # ext=ext.squeeze(1)

        model.train()
        Y_hat=model(X,ext)
        l=loss(Y_hat,Y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    model.eval()
    rmse=0
    cnt=0
    for i,(X,Y,ext) in enumerate(test_dataloader):
        if X.shape[0]!=16:
            continue
        # X = X.squeeze(0)
        # X = X.permute(1, 0, 2, 3)
        # Y = Y.squeeze(0)
        # Y = Y.permute(1, 0, 2, 3)
        # ext = ext.squeeze(0)
        # ext = ext.permute(1, 0, 2)
        # ext = ext.squeeze(1)
        Y_hat=model(X,ext)
        l=loss(Y_hat,Y)
        rmse+=l.item()*len(X)
        cnt+=len(X)
    rmse/=len(test_dataloader)*16
    rmse=np.sqrt(rmse)
    print('epoch=',k+1,' loss=',rmse)

    if rmse<mn_loss:
        mn_loss=rmse
        torch.save(model.state_dict(), os.path.join('paramter', 'urbanz.pt'))
    #     f = open(os.path.join('paramter','results.txt'),'a')
    #     f.write("epoch={}\trmse={:.6f}\n".format(k+1,rmse))
    #     f.close()

    if k%hf==0 and k!=0:
        lr*=0.5
        trainer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))
