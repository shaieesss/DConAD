import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math


class OrdAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(OrdAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear((self.atten_dim * self.head_num), model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = Q.clone()

        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)

class TemporalTransformerBlock(nn.Module):
    def __init__(self,  window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TemporalTransformerBlock, self).__init__()

        self.attention = OrdAttention(window_size, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)
        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        
        return self.norm(x + residual)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTransformerBlock, self).__init__()
        self.attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.attention(x, x, x)
        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        

        return self.norm(x + residual)


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, head_num=4, dropout=0.2):
        super(SpatialTemporalTransformer, self).__init__()
        
        self.time_block = TemporalTransformerBlock(window_size, model_dim, ff_dim, int(model_dim / head_num), head_num, dropout)
        self.feature_block = SpatialTransformerBlock(model_dim, ff_dim, int(model_dim / head_num), head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        # batch, win, d_model
        time_x = self.time_block(x)
        #print(time_x.shape)
        feature_x = self.feature_block(x)
        #print(feature_x.shape)
        
        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm2(x)