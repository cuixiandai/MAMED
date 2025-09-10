import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from bimamba import biMamba, MambaConfig
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
import math
import pdb

icc=64+2
ic=72
oc=256

dp=0.5
ws=13
fs=(ws+1)//2
png_in=fs*fs

L1O=128

num_class=11

d_model=ws*ws

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x):
        return self.conv(x)

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class biMambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(biMambaLayers, self).__init__()
        self.mamba = biMamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x    

class Pooling(nn.Module):
    """
    @article{ref-vit,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  

class ChannelAttn(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8,dropout=0.1):
        super(ChannelAttn, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in,nhead,dim_feedforward,dropout=0.1,norm_first=False)
        self.encoder0 = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=LayerNorm(ic))

    def forward(self, x):
        batch_size, c, h, w = x.shape
        center_h = (h - 1) // 2
        center_w = (w - 1) // 2

        values = x[:, :, center_h, center_w].unsqueeze(1) #.flatten()  # shape: (batch_size * c, )
        values=self.encoder0(values).flatten()

        h_idx = torch.full((batch_size * c,), center_h, device=x.device, dtype=torch.long)
        w_idx = torch.full((batch_size * c,), center_w, device=x.device, dtype=torch.long)

        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(c)
        channel_idx = torch.arange(c, device=x.device).repeat(batch_size)

        x = torch.index_put(x, (batch_idx, channel_idx, h_idx, w_idx), values, accumulate=False)

        return x  
    
def BasicConv(in_channels, out_channels, kernel_size, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),)

class PositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.register_buffer('position_ids', torch.arange(height * width))
        self.embedding = nn.Embedding(height * width, channels)
        self.height = height
        self.width = width

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width

        # [H*W, C]
        pos_embed = self.embedding(self.position_ids)
        # [1, C, H, W]
        pos_embed = pos_embed.view(1, C, H, W)
        # expand to batch
        pos_embed = pos_embed.expand(B, -1, -1, -1)

        return x + pos_embed

class MyModel(torch.nn.Module):
    def __init__(self,batch_size=32,bilinear=True):
        super(MyModel, self).__init__()
        
        self.conv0 = BasicConv(in_channels=icc, out_channels=ic, kernel_size=1, stride=1, padding=0)
        
        self.inc= DoubleConv(ic,oc)
        self.down1= Down(oc,oc*2)
        self.down2=Down(oc*2,oc*4)

        self.up2= Up(oc*2,oc,bilinear)
        self.up1= Up(oc*4,oc*2,bilinear)

        self.dropout=Dropout(dp)

        self.mamba = biMambaLayers(d_model=ic,n_layers=1)

        self.encoder1 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)
        self.encoder2 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)

        self.pos1=PositionalEncoding(ic,ws,ws)

        self.BN=nn.BatchNorm2d(ic)
        self.lmd = nn.Parameter(torch.tensor(0.5))

        self.pool = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_class)

    def forward(self, x):
        x,x1=x
        x=torch.cat((x,x1),1)
        batch_size, c, h, w= x.shape

        x=self.conv0(x)

        x=self.mamba(x)

        x=self.BN(x)
        
        x1=self.pos1(x)

        x=self.encoder1(x1)

        x1=x1.transpose(2,3)
        x1=self.encoder2(x1)
        x1=x1.transpose(2,3)

        x=(1-self.lmd)*x+self.lmd*x1

        x1= self.inc(x)
        
        x2= self.down1(x1)

        x3= self.down2(x2)
      
        x= self.up1(x3,x2)

        x=self.up2(x,x1)
        x=self.dropout(x)

        #x=x.reshape(batch_size,-1,h*w)
        x=x.reshape(batch_size,-1,ws*ws)
        
        x=x.permute(0,2,1)

        x=self.pool(x)

        x = self.classifier(x)

        return x

if __name__=='__main__':
    print('start')
    batch_size=32
    model = MyModel(batch_size=batch_size, bilinear=True)

    #
    dummy_input = torch.randn(batch_size, ic, 13, 13)  

    # 
    output = model(dummy_input)

    # 
    print("Output shape:", output.shape)