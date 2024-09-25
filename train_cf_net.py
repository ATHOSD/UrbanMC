import os.path
import numpy as np
import torch
from torch import nn
from vis import visual
from torch.utils.data import DataLoader
from function import get_dataloader,init_weight,get_cf_dataloader
from model import cf_net

test_cuda = True if torch.cuda.is_available() else False




model=cf_net(num_res=1,in_c=1,out_c=1,num_hid=64)
if test_cuda:
    model=model.cuda()
model.apply(init_weight)
model.load_state_dict(torch.load(os.path.join('paramter','cf_net.pt')))
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)

loss=nn.MSELoss()
if test_cuda:
    loss=loss.cuda()

lr=1e-4
trainer=torch.optim.Adam(model.parameters(),lr=lr)

valid_dataloader=get_cf_dataloader(batch_size=16,path='data',mode='valid',shuffle=True)
train_dataloader=get_cf_dataloader(batch_size=8,path='data',mode='train',shuffle=False)

epochs=300

mn_loss=1e9

for k in range(epochs):
    for i,(X,Y) in enumerate(train_dataloader):
        # model.train()
        # trainer.zero_grad()
        Y_hat = model(X)
        # l = loss(Y_hat, Y)
        # l.backward()
        # trainer.step()

        print(X.size())
        if k==0 and i==0:
            visual(X[0][0], y='fig6.a', z='')
            visual(Y[0][0], y='fig6.b', z='')
            visual(Y_hat[0][0].detach(),y='fig6.c',z='')

    # model.eval()
    # l_sum=0
    # for j,(X,Y) in enumerate(valid_dataloader):
    #     Y_hat=model(X)
    #     l=loss(Y_hat,Y)
    #     l_sum+=l.item()*len(X)
    # rmse=np.sqrt(l_sum/len(valid_dataloader.dataset))
    # print('epoch=', k + 1,' loss=',rmse)
    # if rmse<mn_loss:
    #     mn_loss=rmse
    #     torch.save(model.state_dict(),os.path.join('paramter','cf_net.pt'))



