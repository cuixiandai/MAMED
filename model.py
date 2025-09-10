import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
from typing import Union, Sequence
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

import math
import pdb

icc=64+2
ic=72

oc=ic 

dp=0.5
ws=13 
fs=(ws+1)//2
png_in=fs*fs

L1O=128

num_class=11

d_model=ws*ws

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

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,reverse=False,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.reverse=reverse

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        if self.reverse:
            x=torch.flip(x, dims=[1])

        x = self.encoder(x)

        if self.reverse:
            x=torch.flip(x, dims=[1])

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

class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2)
        return x

class EncoderL(nn.Module):
    """
    Input  : (B, C_in,  H, W)
    Output : (B, C_out, H, W)
    Hidden  : Linear → Transformer → Linear
    """
    def __init__(self,
                 in_channels: int,num_layers=1,
                 nhead: int = 8,
                 dim_feedforward: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * in_channels

        layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True)
        self.layer=TransformerEncoder(layer,num_layers,nn.LayerNorm(in_channels))

        # learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, in_channels))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        x = x.view(B, C_in, -1).permute(0, 2, 1)          # (B, H*W, C_in)

        pos = self.pos_embed[:, :H*W, :].to(x.dtype)
        x = x + pos
        x = self.layer(x)                                 # (B, H*W, C_out)

        x = x.permute(0, 2, 1).view(B, x.size(-1), H, W)  # (B, C_out, H, W)
        return x  

class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[tuple, str],
        res_block: bool = False,
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        # Handle size mismatch with interpolation
        if out.shape[2:] != skip.shape[2:]:
            out = torch.nn.functional.interpolate(
                out, size=skip.shape[2:], mode='nearest'
            )
        out = torch.cat([out, skip], dim=1)
        out = self.conv_block(out)
        return out

class Up_swin(nn.Module):
    def __init__(self, ic,num_layers=2):
        super().__init__()
        self.us = UnetrUpBlock(spatial_dims=2,in_channels=ic*2, out_channels=ic,kernel_size=3,upsample_kernel_size=2,norm_name="batch",res_block=True,)
        
    def forward(self, x,x2):
        x=self.us(x,x2)
        return x

class MyModel(torch.nn.Module):
    def __init__(self,batch_size=32,bilinear=True):
        super(MyModel, self).__init__()
        
        self.conv0 = BasicConv(in_channels=icc, out_channels=ic, kernel_size=1, stride=1, padding=0)
        
        self.inc= DoubleConv(ic,oc)
        self.down1= Down(oc,oc*2)
        self.down2=Down(oc*2,oc*4)
        self.mamba2 = MambaLayers(d_model=oc*2,n_layers=1)
        self.mamba3 = MambaLayers(d_model=oc*4,n_layers=1)

        self.up2= Up_swin(oc)
        self.up1= Up_swin(oc*2)

        self.dropout=Dropout(dp)

        self.mamba1 = MambaLayers(d_model=ic,n_layers=1)
        self.pos1=PositionalEncoding(ic,ws,ws)
        self.attn1 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)
        self.attn2 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)

        self.attn1r = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8,reverse=True)
        self.attn2r = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8,reverse=True)
        self.BN1=nn.BatchNorm2d(ic)

        self.lmd1 = nn.Parameter(torch.tensor(0.25))
        self.lmd2 = nn.Parameter(torch.tensor(0.25))
        self.lmd3 = nn.Parameter(torch.tensor(0.25))

        self.pool = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_class)

    def forward(self, x):
        x,x1=x
        x=torch.cat((x,x1),1)

        x=self.conv0(x)
        batch_size, c, h, w= x.shape

        x=self.mamba1(x)
        x=self.BN1(x)
        x=self.pos1(x)
        x1=self.attn2(x.permute(0,1,3,2)).permute(0,1,3,2)
        x2=self.attn2r(x.permute(0,1,3,2)).permute(0,1,3,2)
        x3=self.attn1r(x)
        x=self.attn1(x)   

        x=self.lmd1*x+self.lmd2*x1+self.lmd3*x2+(1-self.lmd1-self.lmd2-self.lmd3)*x3

        x1= self.inc(x)

        x2= self.down1(x1)
        x2= self.mamba2(x2)
        x3= self.down2(x2)
        x3= self.mamba3(x3)
        x= self.up1(x3,x2)

        x=self.up2(x,x1)

        x=self.dropout(x)

        x=x.reshape(batch_size,-1,h*w)
        
        x=x.permute(0,2,1)

        x=self.pool(x)

        x = self.classifier(x)

        return x

if __name__=='__main__':
    print('start')
    batch_size=32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyModel(batch_size=batch_size, bilinear=True).to(device)

    dummy_input = torch.randn(batch_size, ic, 13, 13).to(device) 

    output = model(dummy_input)

    print("Output shape:", output.shape)