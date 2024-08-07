import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from einops import rearrange

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
class SensorTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model,embed='conv',win_size=100):
        super(SensorTokenEmbedding, self).__init__()
        if embed=='conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                       kernel_size=3, padding=padding, padding_mode='circular', bias=False)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif embed=='linear':
            self.tokenConv=nn.Linear(win_size,d_model)

    def forward(self, x):
        x = self.tokenConv(x)#.transpose(1, 2)
        return torch.tanh(x)
class SensorDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model=1, dropout=0.0,embed='linear',win_size=100):
        super(SensorDataEmbedding, self).__init__()
        # print(embed)
        self.value_embedding = SensorTokenEmbedding(c_in=1, d_model=d_model,embed=embed,win_size=win_size)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B,N,L=x.size()
        #x: batch_size,c_in,window
        x=x[:,:,None,:]#batch_size,c_in,1,window
        # print(x.size())
        x = rearrange(x, 'b c s l -> (b c) s l')
        #x:batch_size*c_in,1,window
        #conv1d
        #value_embed:batch_size*c_in,1,window
        # print(x.size())
        
        x = self.value_embedding(x) #+ self.position_embedding(x)
        # print('embed',x.size())
        x=self.dropout(x)
        #batch*c_in,1,window
        # print('val',x.size())#[200, 1, 51]
        x = torch.reshape(x,(B,N,-1))#rearrange(x, '(b c) s l -> b s l c')
        return x[:,:,:]
