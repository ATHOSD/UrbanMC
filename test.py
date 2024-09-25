import os.path
import torch
import numpy as np
from torch import nn
from model import UrbanZ
import random
from function import get_ft_dataloader,get_mape
from vis import visual

test_cuda = True if torch.cuda.is_available() else False

model=UrbanZ()
if test_cuda:
    model=model.cuda()
model.load_state_dict(torch.load(os.path.join('paramter','urbanz.pt')))

loss=nn.MSELoss()
if test_cuda:
    loss=loss.cuda()

test_dataloader=get_ft_dataloader(path='data',batch_size=16, mode='train',shuffle=False)


def get_mae(x,y):
    return torch.mean(torch.abs(x-y))


def solve(x):
    y=torch.zeros(x.shape[0]//4,x.shape[1]//4)
    if test_cuda:
        y=y.cuda()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i//4][j//4]+=x[i][j]
    return y


def mask(x):
    y=x.clone()
    for i in range(x.shape[0]//4):
        for j in range(x.shape[1]//4):
            rd=random.random()
            if rd<0.75:
                for l in range(4):
                    for r in range(4):
                        y[i*4+l][j*4+r]=2000
    return y


model.eval()
rmse=0
mape=0
mae=0
with open('co2.txt','w') as file:
    for i,(X,Y,ext) in enumerate(test_dataloader):
        if X.shape[0]!=16:
            continue
        Y_hat=model(X,ext)
        l=loss(Y_hat,Y)
        rmse+=l.item()*len(X)
        l2=get_mape(Y_hat,Y)
        mae+=get_mae(Y_hat,Y).item()*len(X)
        mape+=l2.item()*len(X)

        for l in range(Y.shape[0]):
            for r in range(Y.shape[1]):
                val1=Y[l][r][15][15].item()
                val2=Y_hat[l][r][15][15].item()
                file.write(str(val1)+' '+str(val2)+'\n')

    # if i==1:
    #     with open('co1.txt', 'w') as file:
    #         for l in range(Y[0][0].shape[0]):
    #             for r in range(Y[0][0].shape[1]):
    #                 val1=Y[0][0][l][r].item()
    #                 val2=Y_hat[0][0][l][r].item()
    #                 file.write(str(val1)+' '+str(val2)+'\n')

    # for cnt in range(4):
        #     visual(mask(X[0][0]),y='fig7.'+str(cnt),z='')
        # visual(X[0][0],y='fig1.a',z='(a) Coarse-grained urban flows')
        # visual(Y[0][0],y='fig1.b',z='(b) Fine-grained urban flows')
        # visual(solve(Y[0,0,120:128,120:128]),y='fig1.c',annot=True)
        # visual(Y[0,0,120:128,120:128],y='fig1.d',annot=True)

    # if i==0:
    #     visual(X[0][0],y='fc',z='(a) Coarse-grained urban flows')
    #     visual(Y[0][0],y='ff1',z='(b) Fine-grained urban flows')
    #     visual(Y_hat[0][0].detach(),y='ff2',z='(c) UrbanMC fine-grained inference')
# rmse/=len(test_dataloader)*16
# rmse=np.sqrt(rmse)
# print('RMSE=',rmse)
# mae/=len(test_dataloader)*16
# print('MAE=',mae)
# mape/=len(test_dataloader)*16
# print('MAPE=',mape)