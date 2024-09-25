import torch
from torch import nn
from function import transpose_qkv,transpose_output
from timm.models.layers import Mlp, DropPath
from torch.autograd import Variable

test_cuda = True if torch.cuda.is_available() else False


class patch_embedding(nn.Module):

    def __init__(self,img_size=32,patch_size=4,embed_dim=16,in_chans=8):
        super(patch_embedding,self).__init__()
        self.num_patches=(img_size//patch_size)**2
        self.proj=nn.Conv2d(in_channels=in_chans,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.flatten=nn.Flatten(start_dim=2)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,x):  #x: B*C*N*N
        x=self.proj(x)
        x=self.flatten(x).transpose(1,2)
        x=self.norm(x)
        return x




class mask(nn.Module):

    def __init__(self,p=0.8):
        super(mask,self).__init__()
        self.p=p

    def forward(self,x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.p))
        x_masked_all = None
        mask_all = None
        ids_restore_all = None
        for data in range(D):
            noise = torch.rand(N, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x[:, :, data], dim=1, index=ids_keep)
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            if data == 0:
                x_masked_all = x_masked.unsqueeze(-1)
                mask_all = mask.unsqueeze(-1)
                ids_restore_all = ids_restore.unsqueeze(-1)
            else:
                x_masked_all = torch.cat((x_masked.unsqueeze(-1), x_masked_all), dim=-1)
                mask_all = torch.cat((mask_all, mask.unsqueeze(-1)), dim=-1)
                ids_restore_all = torch.cat((ids_restore_all, ids_restore.unsqueeze(-1)), dim=-1)
        return x_masked_all, mask_all, ids_restore_all




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x






class Block(nn.Module):

    def __init__(self,dim,num_heads=4,mlp_ratio=4,qkv_bias=False,drop=0.,attn_drop=0.,norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.ReLU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x





class st_net(nn.Module):

    def __init__(self,in_chans=8,num_heads=4,img_size=32,patch_size=4,p=0.8,num_enc_block=3,num_dec_blocks=1,embed_dim=16):
        super(st_net,self).__init__()

        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))
        self.patch_embed=patch_embedding(in_chans=in_chans,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim)
        self.enc_pos_embed=nn.Parameter(torch.randn(1, int((img_size / patch_size) ** 2)+1, embed_dim))
        self.mask=mask(p=p)
        self.enc_blocks=nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_enc_block)])
        self.enc_norm=nn.LayerNorm(embed_dim)


        self.dec_embed=nn.Linear(embed_dim,embed_dim)
        self.dec_pos_embed=nn.Parameter(torch.randn(1,int((img_size / patch_size) ** 2)+1,embed_dim))
        self.dec_blocks=nn.ModuleList([
            Block(dim=embed_dim,num_heads=1,mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_dec_blocks)])
        self.mask_token=nn.Parameter(torch.randn(1,1,embed_dim))
        self.recover=nn.Sequential(nn.Linear(embed_dim,patch_size**2 *in_chans))
        self.dec_norm=nn.LayerNorm(embed_dim)

        torch.nn.init.normal_(self.cls, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def enc(self,x):
        x=self.patch_embed(x)

        x=x+self.enc_pos_embed[:,1:,:]

        x, _, index = self.mask(x)

        cls_token=self.cls+self.enc_pos_embed[:,:1,:]
        cls_token=cls_token.expand(x.shape[0],-1,-1)
        x=torch.cat((cls_token, x), dim=1)

        for blk in self.enc_blocks:
            x=blk(x)

        x=self.enc_norm(x)
        return x,index


    def dec(self,x,index):
        x=self.dec_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], index.shape[1] +1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=index)
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x=x+self.dec_pos_embed

        for blk in self.dec_blocks:
            x=blk(x)

        x= self.dec_norm(x)

        x=self.recover(x)

        return x[:,1:,:]

    def forward(self,x):
        x,index=self.enc(x)
        x=self.dec(x,index)
        return x




#--------------------------------------------------------cf_net---------------------------------------------------------

class res_net(nn.Module):

    def __init__(self,in_c=64,out_c=64):
        super(res_net,self).__init__()
        self.conv1=nn.Conv2d(in_c,out_c,3,1,1)
        self.bn=nn.BatchNorm2d(out_c)
        self.conv2=nn.Conv2d(out_c,out_c,3,1,1)
        self.relu=nn.ReLU()

    def forward(self,x):
        y=self.conv1(x)
        y=self.bn(y)
        y=self.relu(y)
        y=self.conv2(y)
        y=self.bn(y)
        y+=x
        y=self.relu(y)
        return y




class cf_net(nn.Module):

    def __init__(self,num_res=1,in_c=1,out_c=1,num_hid=64):
        super(cf_net,self).__init__()
        conv_list=[]
        for i in range(num_res+1):
            if i==0:
                conv_list += [nn.Conv2d(in_c, num_hid, 3, 1, 1), nn.ReLU()]
            else:
                conv_list += [res_net(num_hid,num_hid)]
        self.conv_enc=nn.Sequential(*conv_list)

        self.conv_dec=nn.Sequential(nn.Conv2d(num_hid,num_hid,3,1,1),nn.ReLU(),nn.Dropout(0.5))

        up_list=[]
        for i in range(2):
            up_list += [nn.Conv2d(num_hid,num_hid*4,3,1,1),nn.BatchNorm2d(num_hid*4),
                        nn.PixelShuffle(upscale_factor=2),nn.ReLU()]
        self.up=nn.Sequential(*up_list)

        self.conv_out=nn.Sequential(nn.Conv2d(num_hid,out_c,3,1,1),nn.ReLU())

        self.calc=calc(4)

    def forward(self,x):
        y=x
        x=self.conv_enc(x)
        x=self.conv_dec(x)
        x=self.up(x)
        x=self.conv_out(x)
        ans=self.calc(x,y)
        return ans




class calc(nn.Module):

    def __init__(self, up=2):
        super(calc, self).__init__()
        self.up=up
        self.avg=nn.AvgPool2d(up)
        self.up1=nn.Upsample(scale_factor=up, mode='nearest')
        self.up2=nn.Upsample(scale_factor=up, mode='nearest')
        self.eps=1e-4

    def forward(self, x, y):
        out=self.avg(x)*self.up**2
        out=self.up1(out)
        per=torch.div(x,out+self.eps)
        y=self.up2(y)
        return torch.mul(per,y)



#--------------------------------------------------------lt_net---------------------------------------------------------



class lt_net(nn.Module):

    def __init__(self,in_chans=1,num_hid=64):
        super(lt_net,self).__init__()

        self.net=nn.Sequential(nn.Conv2d(in_chans,num_hid,3,1,1),nn.ReLU(),
                               nn.Conv2d(num_hid,num_hid,3,1,1),nn.ReLU())

        self.dec=nn.Sequential(nn.BatchNorm2d(num_hid),nn.AdaptiveAvgPool2d((1,1)),
                               nn.Flatten(),nn.Linear(num_hid,num_hid),nn.ReLU())
        self.alpha=2.0

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self,x):
        x=self.net(x)
        x=self.dec(x)
        return self.normalize(x)*self.alpha



def dis(l,r):
    diff=torch.abs(l-r)
    return torch.pow(diff,2).sum(dim=1)



class triple_loss(nn.Module):
    def __init__(self,alpha=50000):
        super(triple_loss,self).__init__()
        self.alpha=alpha

    def forward(self,a,b,c):
        ans=torch.sum(torch.sqrt(torch.square(a - b)))-torch.sum(torch.sqrt(torch.square(a - c)))
        zero=torch.zeros(size=ans.size())
        ans=max(ans+self.alpha,zero)
        ans=Variable(ans,requires_grad=True)
        return ans




#--------------------------------------------------------urbanz---------------------------------------------------------



class UrbanZ(nn.Module):

    def __init__(self):
        super(UrbanZ,self).__init__()

        self.cf_net=cf_net(num_res=1,in_c=1,out_c=1,num_hid=64)
        self.st_net=st_net(in_chans=16,num_heads=4,img_size=32,patch_size=2,p=0,num_enc_block=4,num_dec_blocks=1,embed_dim=128)
        self.lt_net=lt_net(in_chans=1)
        self.st_up=nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),
                                 nn.PixelShuffle(upscale_factor=2),nn.ReLU(),nn.Conv2d(32,1,3,1,1))

        self.dec=nn.Sequential(nn.Conv2d(129,64,3,1,1),nn.ReLU())

        self.ext1=nn.Embedding(8,4)
        self.ext2=nn.Embedding(24,6)
        self.ext3=nn.Embedding(18,6)
        self.ext=nn.Sequential(nn.Linear(20,128),nn.Dropout(0.5),nn.ReLU(),nn.Linear(128,32*32))

    def forward(self,x,ext):

        e1=self.ext1(ext[:,4].long())
        e2=self.ext2(ext[:,5].long())
        e3=self.ext3(ext[:,6].long())
        e4=ext[:,:4]
        e=torch.cat([e1,e2,e3,e4],dim=1)
        e=self.ext(e).reshape(-1,1,32,32)
        x=e+x

        z=x
        val=x
        y=torch.reshape(x, (-1, 16, 32, 32))
        x=self.cf_net.conv_enc(x)
        x=self.cf_net.conv_dec(x)   #16*64*32*32
        val=self.lt_net.net(val)    #16*64*32*32
        y,_=self.st_net.enc(y)
        y=y[:,1:,:]
        y=y.reshape(1,16,16,-1)
        y=y.permute(0,3,1,2)
        y=self.st_up(y)   #1*1*32*32

        y=y.repeat(16,1,1,1)
        x=torch.cat([x,y,val],dim=1)
        x=self.dec(x)

        x=self.cf_net.up(x)
        x=self.cf_net.conv_out(x)
        ans=self.cf_net.calc(x,z)
        return ans






# class patch_embedding3d(nn.Module):
#
#     def __init__(self,batch_size=16,img_size=32,patch_size=4,embed_dim=16,dropout=0.0):
#         super(patch_embedding3d,self).__init__()
#         self.num_patches=int((img_size/patch_size)**2)*batch_size
#         self.batch_size=batch_size
#         self.img_size=img_size
#         self.patch_size=patch_size
#         patch_dim=patch_size*patch_size
#         self.linear=nn.Linear(patch_dim,embed_dim)
#         self.pos_embed=nn.Parameter(torch.randn(self.num_patches,embed_dim,requires_grad=False))
#         self.dropout=nn.Dropout(dropout)
#
#     def forward(self,x):             #x:B*1*N*N
#         x=torch.transpose(x,0,1)
#         x=torch.squeeze(x,0)
#         tmp=int(self.img_size/self.patch_size)
#         x=x.reshape(self.batch_size,tmp,self.patch_size,tmp,self.patch_size)
#         x=x.permute(0,1,3,2,4)
#         x=x.reshape(self.num_patches,-1)
#         x=self.linear(x)
#         x+=self.pos_embed
#         x=self.dropout(x)
#         return x




# class mask(nn.Module):
#
#     def __init__(self,num_patches,p=0.8):
#         super(mask,self).__init__()
#         self.p=p
#         self.index=[i for i in range(num_patches)]
#         random.shuffle(self.index)
#         self.mask_len=int(num_patches*p)
#         self.unmask_len=num_patches-self.mask_len
#         self.mask_index=self.index[:self.mask_len]
#         self.unmask_index=self.index[self.mask_len:]
#
#     def forward(self,x):
#         return self.unmask_index,x[self.unmask_index]




# class multihead_attention(nn.Module):
#
#     def __init__(self,num_heads,num_hid,query_size,key_size,value_size,bias=False,dropout=0.0):
#         super(multihead_attention,self).__init__()
#         self.num_heads = num_heads
#         self.attention = d2l.DotProductAttention(dropout)
#         self.W_q=nn.Linear(query_size, num_hid, bias=bias)
#         self.W_k=nn.Linear(key_size, num_hid, bias=bias)
#         self.W_v=nn.Linear(value_size, num_hid, bias=bias)
#         self.W_o=nn.Linear(num_hid, num_hid, bias=bias)
#
#     def forward(self,q,k,v):
#         q=transpose_qkv(self.W_q(q),num_heads=self.num_heads)
#         k=transpose_qkv(self.W_k(k), num_heads=self.num_heads)
#         v=transpose_qkv(self.W_v(v), num_heads=self.num_heads)
#         output=self.attention(q,k,v)
#         ans=transpose_output(output,num_heads=self.num_heads)
#         return ans




# class transformer_block(nn.Module):
#
#     def __init__(self,embed_dim,num_heads,bias=False,dropout_attn=0.0,dropout_linear=0.0):
#         super(transformer_block,self).__init__()
#         self.norm1=nn.LayerNorm(embed_dim)
#         self.mha=multihead_attention(num_heads=num_heads,num_hid=embed_dim,query_size=embed_dim,key_size=embed_dim,
#                                      value_size=embed_dim,bias=bias,dropout=dropout_attn)
#         self.norm2=nn.LayerNorm(embed_dim)
#         self.mlp=nn.Sequential(nn.Linear(embed_dim,embed_dim*4),nn.GELU(),nn.Dropout(dropout_linear),
#                                nn.Linear(embed_dim*4,embed_dim),nn.Dropout(dropout_linear))
#
#     def forward(self,x):
#         x=x+self.mha(self.norm1(x),self.norm1(x),self.norm1(x))
#         x=x+self.mlp(self.norm2(x))
#         return x




# class recover_pos(nn.Module):
#
#     def __init__(self,embed_dim=16,batch_size=16,img_size=32,patch_size=4):
#         super(recover_pos,self).__init__()
#         self.embed_dim=embed_dim
#         self.batch_size=batch_size
#         self.img_size=img_size
#         self.patch_size=patch_size
#
#     def forward(self,x):
#         tmp=int(self.img_size/self.patch_size)
#         x=x.reshape(self.batch_size,tmp,tmp,self.patch_size,self.patch_size)
#         x=torch.transpose(x,2,3)
#         x=x.reshape(self.batch_size,tmp,self.patch_size,-1)
#         x=x.reshape(self.batch_size,-1,self.img_size)
#         x=torch.unsqueeze(x,1)
#         return x


# class st_net(nn.Module):
#
#     def __init__(self,num_heads,img_size=32,batch_size=16,patch_size=4,p=0.8,num_trans_block=4,embed_dim=16):
#         super(st_net,self).__init__()
#         self.embed=patch_embedding3d(batch_size=batch_size,patch_size=patch_size,
#                                      dropout=0.5,img_size=img_size,embed_dim=embed_dim)
#         self.mask=mask(num_patches=batch_size*int((img_size/patch_size)**2),p=p)
#         trans_list=[]
#         for i in range(num_trans_block):
#             trans_list+=[transformer_block(embed_dim=embed_dim,num_heads=num_heads,bias=False,
#                                            dropout_attn=0.5,dropout_linear=0.5)]
#         self.trans=nn.Sequential(*trans_list)
#         self.num_patches=batch_size*int((img_size/patch_size)**2)
#         self.re=recover_pos(embed_dim=embed_dim,batch_size=batch_size,img_size=img_size,patch_size=patch_size)
#
#         dec_list=[]
#         dec_list+=[nn.Linear(embed_dim,10)]
#         for i in range(1):
#             dec_list+=[transformer_block(embed_dim=10,num_heads=1,bias=False,dropout_attn=0.2,dropout_linear=0.3)]
#         dec_list+=[nn.LayerNorm(10),nn.Linear(10,patch_size*patch_size),self.re]
#         self.dec=nn.Sequential(*dec_list)
#
#     def forward(self,x):
#         x=self.embed(x)
#         total_len=x.shape[0]
#         unmask_index,x=self.mask(x)
#         x=self.trans(x)
#         recover=torch.zeros(size=(total_len,x.shape[1]))
#         if test_cuda:
#             recover=recover.cuda()
#         recover[unmask_index]=x
#         output=self.dec(recover)
#         return output