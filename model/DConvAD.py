import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import SpatialTemporalTransformer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten
#from NF import MAF

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list

class FF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)

class DConvAD(nn.Module):
    def __init__(self, win_size, enc_in, d_model=256, e_layers=1, dropout=0.0, output_attention=True, in_channel=6):
        super(DConvAD, self).__init__()
        self.output_attention = output_attention
        self.win_size = win_size
        
        self.in_channel = in_channel
        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        self.x_dependency_learning = nn.Sequential() 
        self.d_x_dependency_learning = nn.Sequential()
        for l in range(e_layers):
            self.x_dependency_learning.add_module(f"x{l}", SpatialTemporalTransformer(ff_dim=d_model, window_size=self.win_size, model_dim=d_model))
            self.d_x_dependency_learning.add_module(f"d_x{l}", SpatialTemporalTransformer(ff_dim=d_model, window_size=(self.win_size-1), model_dim=d_model))

        self.x_dependency = nn.Bilinear(win_size-1, win_size, win_size)
        self.x_dependency_t = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(win_size, win_size),
            nn.Sigmoid()
        ) 
        self.d_x_dependency = FF(2 * win_size-1, win_size)

    def forward(self, x, d_x):
        B, L, M = x.shape #Batch win_size channel
        
        revin_layer = RevIN(num_features=M)
        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        d_x = revin_layer(d_x, 'norm')
        
        # embedding
        # batch, win, d_model
        x_ori = self.embedding_window_size(x)
        # batch, win-1, d_model
        d_x_ori =self.embedding_window_size(d_x)
        
        # dependency learning
        # batch, win, d_model
        x_hidden = self.x_dependency_learning(x_ori)
        # batch, win-1, d_model
        d_x_hidden = self.d_x_dependency_learning(d_x_ori)
        
        # batch, d_model, win
        x_temp = x_hidden.permute(0, 2, 1)
        # batch, d_model, win-1
        d_x_temp = d_x_hidden.permute(0, 2, 1)

        bi_dependency = self.x_dependency(d_x_temp, x_temp)
        bi_dependency_t = self.x_dependency_t(bi_dependency).permute(0, 2, 1)
        cat_dependency = self.d_x_dependency(torch.cat((d_x_temp, x_temp), dim=-1)).permute(0, 2, 1)
        
        return [bi_dependency_t.unsqueeze(dim=1)], [cat_dependency.unsqueeze(dim=1)]
        

