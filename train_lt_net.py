import torch
import os
from model import lt_net,triple_loss,dis
from function import create_lt_data,get_lt_dataloader

test_cuda = True if torch.cuda.is_available() else False
#create_lt_data(mode='train',cnt=5)
#create_lt_data(mode='valid',cnt=5)

model=lt_net(in_chans=1)

train_dataloader=get_lt_dataloader(batch_size=8,path='data',mode='train',shuffle=True,cnt=8)
valid_dataloader=get_lt_dataloader(batch_size=8,path='data',mode='valid',shuffle=True,cnt=8)

lr=1e-4
trainer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))

loss=triple_loss()

if test_cuda:
    model=model.cuda()
    loss=loss.cuda()

epoch=200

mn_loss=1e9

alpha=8.0

for i in range(epoch):
    l_sum=0
    tmp=0
    for k,(X,Y,Z) in enumerate(train_dataloader):
        model.train()
        x=model(X)
        y=model(Y)
        z=model(Z)
        l=torch.clamp(alpha+dis(x,y)-dis(x,z),min=0.0).mean()
        trainer.zero_grad()
        tmp+=l.item()*len(X)
        l.backward()
        trainer.step()

    for k,(X,Y,Z) in enumerate(valid_dataloader):
        model.eval()
        x=model(X)
        y=model(Y)
        z=model(Z)
        l=torch.clamp(alpha+dis(x,y)-dis(x,z),min=0.0).mean()
        l_sum+=l.item()*len(X)
    l_sum/=len(valid_dataloader)
    tmp/=len(train_dataloader)
    if l_sum<mn_loss:
        mn_loss=l_sum
        torch.save(model.state_dict(),os.path.join('paramter','lt_net2.pt'))
    print('epoch=',i+1,' loss=',l_sum,tmp)
