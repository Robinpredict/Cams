import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import SensorAnomalyAttentionLayer,SensorAttention,GlobalAttention,GlobalAttention2,SensorAnomalyAttention#AnomalyAttention, AttentionLayer,SensorAttention,GlobalAttention,GAttentionLayer
from .embed import DataEmbedding, TokenEmbedding,SensorDataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, globalx,attn_mask=None):
        #x:batch,nsensors,seq_len
        #seq_len:d_model
        #global_series is same, no need to return all
        #how to select series attention is important
        new_x, attn,global_out,global_series= self.attention(
            x, x, x,globalx,globalx,globalx,
            attn_mask=attn_mask
        )
        #newx:batch,nsensors,seq_len
        # print(new_x.size(),x.size())
        new_x=self.dropout(new_x)#
        global_out=self.dropout(global_out)
        # new_x=rearrange(new_x, 'b s l -> b l s')
        x =  new_x#self.dropout(new_x)
        # x=rearrange(new_x, 'b l s -> b s l')
        # print(x.size())
        y = x = self.norm1(x)
        # print(y.size())
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        
        globaly = globalx = self.norm1(global_out)
        # print(y.size())
        globaly = self.dropout(self.activation(self.conv1(globaly.transpose(-1, 1))))
        globaly = self.dropout(self.conv2(globaly).transpose(-1, 1))

        return self.norm2(x + y), attn,self.norm2(globalx+globaly),global_series#, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None,global_norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.global_norm=global_norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        local_series_list = []
        global_series_list = []
        # sigma_list = []
        globalx=x
        for attn_layer in self.attn_layers:
            x, local_series, globalx,global_series= attn_layer(x,globalx, attn_mask=attn_mask)
            local_series_list.append(local_series)
            global_series_list.append(global_series)
            # prior_list.append(prior)
            # sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)
            globalx=self.global_norm(globalx)

        return x, local_series_list,globalx,global_series_list#, prior_list, sigma_list




class DualGlobalSensor(nn.Module):
    def __init__(self, win_size, enc_in, c_out, forecast_step=1,d_model=512, n_heads=8, e_layers=3, d_ff=512,idx=[51],
                 dropout=0.0, activation='gelu', output_attention=True,n_group=5,nsensor=51,device = torch.device("cuda:1"),task='C',args=None):
        super(DualGlobalSensor, self).__init__()
        self.output_attention = output_attention
        self.n_group=n_group
        self.device=device
        self.task=task
        # Encoding
        self.embedding =SensorDataEmbedding(enc_in, enc_in, dropout)

        # Encoder
        # win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, output_attention=False,
        #              n_group=5, nsensor=5, n_heads=1,device=torch.device('cuda:0')):
        self.encoder = Encoder(
            [
                EncoderLayer(
                    SensorAnomalyAttentionLayer(
                        SensorAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention,
                                        n_group=self.n_group[l], 
                                        nsensor=nsensor[l],#nsensor, 
                                        # idx=args.idx,
                                        n_heads=n_heads,device=self.device),
                        GlobalAttention2(win_size, False, attention_dropout=dropout, output_attention=output_attention,
                                        n_group=self.n_group[l], 
                                        nsensor=nsensor[l],#nsensor, 
                                        # idx=args.idx,
                                        n_heads=n_heads,device=self.device),
                        d_model, n_heads,
                        idx=idx[l],
                        n_group=self.n_group[l]),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            global_norm_layer=torch.nn.LayerNorm(d_model)
        )
        if self.task=='Forecast':
            #B,ngroup,dmodel
            self.projection1 =  nn.Conv1d(in_channels=self.n_group[e_layers-1],
                                                 out_channels=enc_in,
                                                 kernel_size=1,
                                                 bias=True)
            self.projection2 = nn.Conv1d(in_channels=d_model,
                                                 out_channels=forecast_step,
                                                 kernel_size=1,
                                                 bias=True)
            # self.projection=nn.Sequential(projection1,
                                          # nn.ReLU(),
                                          # projection2
                                           # )
        elif self.task=='Encode':
            self.projection=nn.ModuleList()
            # print(d_model,c_out,d_model*self.n_group[-1])
            projection1=nn.Linear(d_model*self.n_group[-1],c_out,bias=True)
            projection2=nn.Linear(c_out,c_out,bias=True)
            self.projection=nn.Sequential(projection1,
                                          nn.ReLU(),
                                          projection2,
                
                                      )
        else:
            self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        #batch_size,nsensor,window
        enc_out = self.embedding(x)
        #batch_size,nsensor,dmodel
        # print('enc_out',enc_out.size())
        enc_out,sensor_assoc,global_out,global_assoc  = self.encoder(enc_out)
       
        # print('global enc_out',enc_out.size())
        if self.task=='Forecast':
            p1=torch.relu(self.projection1(enc_out))
            #B,sensor,seq_len
            # print('p1',p1.size())
            p1=torch.transpose(p1,1,2)
            #B,seq_len,sensor
            # print('pt',p1.size())
            enc_out=self.projection2(p1)
            enc_out=torch.transpose(enc_out,1,2)
            
            gp=torch.relu(self.projection1(global_out))
            #B,sensor,seq_len
            # print('p1',p1.size())
            gp=torch.transpose(gp,1,2)
            #B,seq_len,sensor
            # print('pt',p1.size())
            global_out=self.projection2(gp)
            global_out=torch.transpose(global_out,1,2)
            
        elif self.task=='Encode':
            # print('global enc_out',enc_out.size())
            enc_out=rearrange(enc_out, 'b g l -> b (g l)')
            # print('enc_out',enc_out.size())
            enc_out=self.projection(enc_out)
            global_out=rearrange(global_out, 'b g l -> b (g l)')
            # print('enc_out',enc_out.size())
            global_out=self.projection(global_out)
        else:
            enc_out = self.projection(enc_out)
            global_out=self.projection(global_out)
        #B,G,d_model
        # print('project enc_out',enc_out.size())
        if self.output_attention:
            return enc_out, sensor_assoc,global_out, global_assoc#, prior, sigmas
        else:
            return enc_out,global_out  # [B, L, D]

# enc_in=51
# dropout=0
# window=100
# ts=torch.rand(2,window,enc_in)
# embedding = SensorDataEmbedding(enc_in, enc_in, dropout)
# from easydict import EasyDict
# args=EasyDict()

# args.enc_in=51
# args.dropout=0
# args.window=100
# args.win_size=args.window
# args.d_model=args.window
# # args.ts=norm_ts#=torch.rand(2,enc_in,window)
# args.n_group=[80,20,10]
# args.nsensor=[args.enc_in]
# if len(args.n_group)>1:
#     args.nsensor.extend(args.n_group[:-1])
# args.task='Forecast'
# args.forecast_step=1
# args.dataset='SWaT'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# args.device=device
# args.n_heads=1
# args.e_layers=3
# args.lr=1e-4#hyper[3]
# args.c_out=20


# model=DualGlobalSensor(win_size=args.win_size, enc_in=args.enc_in, c_out=args.c_out, 
#                             d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_ff=512,
#               dropout=0, activation='gelu', output_attention=True,
#               n_group=args.n_group,nsensor=args.nsensor,device = args.device,
#               task=args.task,args=args)

# x=torch.rand(2,args.window,args.enc_in)
# enc_out, sensor_assoc,global_out, global_assoc=model(x)
