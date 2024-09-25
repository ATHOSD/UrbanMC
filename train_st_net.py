import os

import numpy as np
import torch
from torch import nn
from model import st_net
from function import get_dataloader,init_weight,patch,get_st_dataloader

test_cuda = True if torch.cuda.is_available() else False

patch_size=2
batch_size=64

model=st_net(in_chans=16,num_heads=4,img_size=32,patch_size=patch_size,p=0.75,num_enc_block=4,num_dec_blocks=1,embed_dim=128)
if test_cuda:
    model=model.cuda()
model.apply(init_weight)

valid_dataloader=get_dataloader(batch_size=batch_size,path='data',mode='valid',shuffle=False)
train_dataloader=get_st_dataloader(batch_size=batch_size,path='data',mode='train',shuffle=False)

loss=nn.MSELoss()
if test_cuda:
    loss=loss.cuda()

lr=0.001
trainer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))

epochs=200


def get_loss(x,y):
    ls=(x-y)**2
    ls=ls.mean(dim=-1)
    ls=ls.sum()
    return ls

mn_loss=1e9


for k in range(epochs):
    for i,(X,_) in enumerate(train_dataloader):
        X=X.squeeze(0)
        model.train()
        Y_hat=model(X)
        X_pat=patch(X,patch_size=patch_size)
        trainer.zero_grad()
        l=get_loss(Y_hat, X_pat)
        l.backward()
        trainer.step()

    model.eval()
    l_sum=0
    cnt=0
    for j,(X, Y) in enumerate(valid_dataloader):
        if X.shape[0] != batch_size:
            continue
        X = X.reshape(-1, 16, 32, 32)
        Y_hat=model(X)
        cnt+=1
        X_pat = patch(X, patch_size=patch_size)
        l=get_loss(Y_hat,X_pat)
        l_sum+=l.item()/Y_hat.shape[0]/Y_hat.shape[1]
    rmse=np.sqrt(l_sum /cnt)
    print('epoch=', k + 1, ' loss=', rmse)
    if rmse < mn_loss:
        mn_loss = rmse
        torch.save(model.state_dict(), os.path.join('paramter', 'st_net3.pt'))
