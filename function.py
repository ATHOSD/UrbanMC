import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


test_cuda = True if torch.cuda.is_available() else False



def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb


def patch(x,patch_size=4):
    B,C,N=x.shape[0],x.shape[1],x.shape[2]
    n=N//patch_size
    y=x.reshape(shape=(B,C,n,patch_size,n,patch_size))
    y=y.permute(0,2,4,3,5,1)
    y=y.reshape(shape=(B,n*n,C*patch_size*patch_size))
    return y


def get_dataloader(batch_size,path,mode,shuffle=True):
    path=os.path.join(path,mode)
    arr_X=np.load(os.path.join(path,'X.npy'))
    arr_Y=np.load(os.path.join(path,'Y.npy'))
    X=torch.Tensor(arr_X)
    Y=torch.Tensor(arr_Y)
    X=torch.unsqueeze(X,1)
    Y=torch.unsqueeze(Y,1)
    if test_cuda:
        X=X.cuda()
        Y=Y.cuda()
    data=torch.utils.data.TensorDataset(X,Y)
    dataloader=DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_cf_dataloader(batch_size,path,mode,shuffle=True):
    path=os.path.join(path,mode)
    arr_X=np.load(os.path.join(path,'X.npy'))
    arr_Y=np.load(os.path.join(path,'X.npy'))
    X=torch.Tensor(arr_X)
    Y=torch.Tensor(arr_Y)
    X=torch.unsqueeze(X,1)
    Y=torch.unsqueeze(Y,1)
    Z=torch.zeros(X.shape[0],1,8,8)
    for k in range(X.shape[0]):
        for i in range(8):
            for j in range(8):
                l=4*i
                r=4*j
                for p in range(4):
                    for q in range(4):
                        Z[k,0,i,j]+=X[k,0,l+p,r+q]
    if test_cuda:
        Z=Z.cuda()
        Y=Y.cuda()
    data=torch.utils.data.TensorDataset(Z,Y)
    dataloader=DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_st_dataloader(batch_size,path,mode,shuffle=True):
    path = os.path.join(path, mode)
    arr_X = np.load(os.path.join(path, 'X.npy'))
    X = torch.Tensor(arr_X)
    if test_cuda:
        X = X.cuda()
    len = X.shape[0]
    ans=torch.zeros(len-batch_size,batch_size//16,16,32,32)
    for i in range(len-batch_size):
        tmp=X[i:i+batch_size,:,:].clone()
        tmp=tmp.reshape(-1,16,32,32)
        ans[i,:,:,:,:]=tmp
    if test_cuda:
        ans=ans.cuda()
    data=torch.utils.data.TensorDataset(ans,ans)
    dataloader=DataLoader(data,batch_size=1,shuffle=shuffle)
    return dataloader


def get_ft_dataloader(batch_size,path,mode,shuffle=True):
    path=os.path.join(path,mode)
    arr_X=np.load(os.path.join(path,'X.npy'))
    arr_Y=np.load(os.path.join(path,'Y.npy'))
    arr_Z=np.load(os.path.join(path,'ext.npy'))
    X=torch.Tensor(arr_X)
    Y=torch.Tensor(arr_Y)
    Z=torch.Tensor(arr_Z)
    X=torch.unsqueeze(X,1)
    Y=torch.unsqueeze(Y,1)
    if test_cuda:
        X=X.cuda()
        Y=Y.cuda()
        Z=Z.cuda()
    data=torch.utils.data.TensorDataset(X,Y,Z)
    dataloader=DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_ft2_dataloader(batch_size,path,mode,shuffle=True):
    path = os.path.join(path, mode)
    arr_X = np.load(os.path.join(path, 'X.npy'))
    arr_Y = np.load(os.path.join(path, 'Y.npy'))
    arr_Z = np.load(os.path.join(path, 'ext.npy'))
    X = torch.Tensor(arr_X)
    Y = torch.Tensor(arr_Y)
    Z = torch.Tensor(arr_Z)
    if test_cuda:
        X = X.cuda()
        Y = Y.cuda()
        Z = Z.cuda()
    len = X.shape[0]
    ans_X=torch.zeros(len-batch_size,batch_size//16,16,32,32)
    ans_Y=torch.zeros(len-batch_size,batch_size//16,16,128,128)
    ans_Z=torch.zeros(len-batch_size,batch_size//16,16,7)
    for i in range(len-batch_size):
        tmp=X[i:i+batch_size,:,:].clone()
        tmp=tmp.reshape(-1,16,32,32)
        tmp_Y=Y[i:i+batch_size,:,:].clone()
        tmp_Z=Z[i:i+batch_size,:].clone()
        ans_X[i,:,:,:,:]=tmp
        ans_Y[i,:,:,:,:]=tmp_Y.reshape(-1,16,128,128)
        ans_Z[i,:,:,:]=tmp_Z.reshape(-1,16,7)
    if test_cuda:
        ans_X=ans_X.cuda()
        ans_Y=ans_Y.cuda()
        ans_Z=ans_Z.cuda()
    data=torch.utils.data.TensorDataset(ans_X,ans_Y,ans_Z)
    dataloader=DataLoader(data,batch_size=1,shuffle=shuffle)
    return dataloader


def transpose_qkv(X,num_heads):  #X: num_qkv * dim
    X=X.reshape(X.shape[0],num_heads,-1) #X: num_qkv * num_heads * dim/num_heads
    X=X.permute(1,0,2)  #X: num_head * num_qkv * dim/num_heads
    return X



def transpose_output(X,num_heads):
    X=X.permute(1,0,2)
    return X.reshape(X.shape[0],-1)


def init_weight(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')


def get_mape(pred,real):
    ori_real=real.clone()
    if test_cuda:
        ori_real=ori_real.cuda()
    real[real==0]=1
    return torch.mean(torch.abs(ori_real - pred)/real)


def dis(l,r):
    l=torch.Tensor(l)
    r=torch.Tensor(r)
    return torch.sqrt(torch.sum(torch.square(l-r)))


def create_lt_data(mode,cnt=8):
    path=os.path.join('data',mode)
    X=np.load(os.path.join(path,'X.npy'))
    k=len(X)
    anchor=[]
    posi=[]
    nega=[]
    for i in range(k):
        l=X[i]
        d=[0 for _ in range(k)]
        vis=[0 for _ in range(k)]
        for j in range(k):
            if i == j:
                continue
            r=X[j]
            d[j]=dis(l,r)
        for o in range(cnt):
            mn=-1
            mx=-1
            for j in range(k):
                if vis[j]==1 or j==i:
                    continue
                if mn==-1 or d[mn]>d[j]:
                    mn=j
                if mx==-1 or d[mx]<d[j]:
                    mx=j
            vis[mn]=1
            vis[mx]=1
            anchor.append(X[i])
            posi.append(X[mn])
            nega.append(X[mx])

    anchor=np.array(anchor)
    posi=np.array(posi)
    nega=np.array(nega)
    anchor_path=os.path.join(path, str(cnt)+'anchor.npy')
    posi_path=os.path.join(path, str(cnt)+'posi.npy')
    nega_path=os.path.join(path, str(cnt)+'nega.npy')
    np.save(anchor_path,anchor)
    np.save(posi_path,posi)
    np.save(nega_path,nega)


def get_lt_dataloader(batch_size,path,mode,shuffle=True,cnt=8):
    path=os.path.join(path,mode)
    arr_anchor=np.load(os.path.join(path,str(cnt)+'anchor.npy'))
    arr_posi=np.load(os.path.join(path,str(cnt)+'posi.npy'))
    arr_nega=np.load(os.path.join(path, str(cnt)+'nega.npy'))
    anchor=torch.Tensor(arr_anchor)
    posi=torch.Tensor(arr_posi)
    nega=torch.Tensor(arr_nega)
    anchor=torch.unsqueeze(anchor,1)
    posi=torch.unsqueeze(posi,1)
    nega=torch.unsqueeze(nega,1)
    if test_cuda:
        anchor=anchor.cuda()
        posi=posi.cuda()
        nega=nega.cuda()
    data=torch.utils.data.TensorDataset(anchor,posi,nega)
    dataloader=DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader