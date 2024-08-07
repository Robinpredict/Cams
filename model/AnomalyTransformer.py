import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import AnomalyAttention, AttentionLayer,SensorAttention,GlobalAttention,GAttentionLayer
from .embed import DataEmbedding, TokenEmbedding,SensorDataEmbedding
from .RevIN import RevIN

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

    def forward(self, x,attn_mask=None):
        #x:batch,nsensors,seq_len
        #seq_len:d_model
        new_x, attn= self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        #newx:batch,nsensors,seq_len
        # print(new_x.size(),x.size())
        new_x=self.dropout(new_x)#
        # new_x=rearrange(new_x, 'b s l -> b l s')
        x =  new_x#self.dropout(new_x)
        # x=rearrange(new_x, 'b l s -> b s l')
        # print(x.size())
        y = x = self.norm1(x)
        # print(y.size())
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn#, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            # prior_list.append(prior)
            # sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list#, prior_list, sigma_list
class Reconstruct(nn.Module):
    def __init__(self,d_model,c_out,ngroup=5,nsensor=51,R_arch='Linear'):
        super(Reconstruct, self).__init__()
        # self.projection1=nn.Linear(d_model, c_out, bias=True)
        # self.projection2=nn.Linear(ngroup,nsensor,bias=True)
        self.R_arch=R_arch
        if  self.R_arch=='Linear':
            self.projection1 = nn.Conv1d(in_channels=ngroup, out_channels=nsensor, kernel_size=1)
            self.projection2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1)
            # self.activation = F.relu
        elif self.R_arch=='LSTM':
            #input B,G,d_model
            self.projection1=nn.LSTM(input_size=ngroup,hidden_size=nsensor,num_layers=1,batch_first=True)
            self.projection2=nn.Linear(d_model,c_out)
    def forward(self,x):
        #x:B,G,d_model
        # x=rearrange(x,'B G L->B L G')
        # print(x.size(),self.R_arch)
        if self.R_arch=='Linear':
            x=self.projection1(x)
            # x=self.activation(x)
            #x:B,G,L
            x=torch.sigmoid(x)
            x=rearrange(x,'B G L->B L G')
            x=self.projection2(x)
            x=rearrange(x,'B L S->B S L')
        elif self.R_arch=='LSTM':
            x=rearrange(x,'B G L->B L G')
            x,_=self.projection1(x)
            # print(x.shape)
            x=rearrange(x,'B L H->B H L')
            x=self.projection2(x)
            #x B.L.cout
        return x

class ADSensor(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=False,n_group=5,nsensor=51,
                 forecast_step=1,device = torch.device("cuda:0"),task='C',R_arch='Linear',
                 revin=True,affine = True, subtract_last = False,output_encoding=False,sensor_embed='linear',
                 args=None):
        super(ADSensor, self).__init__()
        self.sensor_embed=sensor_embed
        self.output_attention = output_attention
        self.n_group=n_group
        self.device=device
        self.task=task
        self.forecast_step=forecast_step
        self.revin = revin
        self.output_encoding=output_encoding
        self.R_arch=R_arch
        
        self.direct_dim=0
        # c_out=
        if self.direct_dim>0:
                self.direct_link = nn.Linear(enc_in,self.direct_dim)
                self.n_group[-1]=self.n_group[-1]-self.direct_dim
        
        if self.revin: self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)
        # print(self.sensor_embed)
        # Encoding
        self.embedding =SensorDataEmbedding(enc_in, d_model, dropout,embed=self.sensor_embed,win_size=win_size)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        SensorAttention(d_model, #win_size
                                        False, attention_dropout=dropout, output_attention=output_attention,
                                        n_group=self.n_group[l], 
                                        nsensor=nsensor[l],#nsensor, 
                                        n_heads=n_heads,device=self.device),
                        
                        
                        d_model, n_heads,
                        n_group=self.n_group[l]),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if self.task=='Forecast':
            #B,ngroup,dmodel
            self.projection1 =  nn.Conv1d(in_channels=self.n_group[e_layers-1],
                                                 out_channels=enc_in,
                                                 kernel_size=1,
                                                 bias=True)
            self.projection2 = nn.Conv1d(in_channels=d_model,
                                                 out_channels=self.forecast_step,
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
            # self.projection = nn.Linear(d_model, c_out, bias=True)
            #batch, ngroup,c_out
            self.projection=Reconstruct(d_model,c_out,ngroup=self.n_group[-1],nsensor=nsensor[0]-self.direct_dim,R_arch=self.R_arch)

    def forward(self, x):
        #batch_size,nsensor,window
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        #revin
        #patching
        if self.direct_dim>0:
            direct_x = self.direct_link(x.permute(0,2,1)).permute(0,2,1)#B,window,direct_dim
        enc_out = self.embedding(x)
        # print(enc_out.shape)
        #batch_size,nsensor,dmodel
        # print('enc_out',enc_out.size())
        enc_out,sensor_assoc = self.encoder(enc_out)
        feats=enc_out
        # print('Sensor enc_out',feats.size())
        if self.task=='Forecast':
            p1=torch.relu(self.projection1(enc_out))
            # print('p1',p1.size())
            p1=torch.transpose(p1,1,2)
            # print('pt',p1.size())
            enc_out=self.projection2(p1)
            enc_out=torch.transpose(enc_out,1,2)
        elif self.task=='Encode':
            # print('global enc_out',enc_out.size())
            enc_out=rearrange(enc_out, 'b g l -> b (g l)')
            # print('enc_out',enc_out.size())
            enc_out=self.projection(enc_out)
        
        else:
            enc_out = self.projection(enc_out)
        #B,G,d_model
        # print('project enc_out',enc_out.size())
        if self.revin: 
            enc_out = enc_out.permute(0,2,1)
            enc_out = self.revin_layer(enc_out, 'denorm')
            enc_out = enc_out.permute(0,2,1)
        # direct_x : B,window,direct_dim
        # print(enc_out.size(),direct_x.size())
        if self.direct_dim>0:
            enc_out=torch.cat([enc_out,direct_x],dim=1)
        if self.output_attention:
            return enc_out, sensor_assoc#, prior, sigmas
        elif self.output_encoding:
            return enc_out,feats
        else:
            return enc_out  # [B, L, D]
        
class GlobalSensor(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True,n_group=5,nsensor=51,forecast_step=1,R_arch='Linear',
                 device = torch.device("cuda:0"),task='C',args=None):
        super(GlobalSensor, self).__init__()
        self.output_attention = output_attention
        self.n_group=n_group
        self.device=device
        self.task=task
        self.R_arch=R_arch
        self.forecast_step=forecast_step
        # Encoding
        # self.embedding =SensorDataEmbedding(enc_in, enc_in, dropout)
        self.embedding =SensorDataEmbedding(enc_in, d_model, dropout,embed='linear',win_size=win_size)
        self.idx=[torch.arange(i).to(device) for i in nsensor]
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    GAttentionLayer(
                        GlobalAttention(d_model, False, attention_dropout=dropout, output_attention=output_attention,
                                        n_group=self.n_group[l], 
                                        nsensor=nsensor[l],#nsensor, 
                                        # idx=args.idx,
                                        n_heads=n_heads,device=self.device),
                        d_model, n_heads,
                        idx=self.idx[l],
                        n_group=self.n_group[l]),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if self.task=='Forecast':
            #B,ngroup,dmodel
            # self.projection1 =  nn.Conv1d(in_channels=self.n_group[e_layers-1],
            #                                      out_channels=enc_in,
            #                                      kernel_size=1,
            #                                      bias=True)
            # self.projection2 = nn.Conv1d(in_channels=d_model,
            #                                      out_channels=self.forecast_step,
            #                                      kernel_size=1,
            #                                      bias=True)
            self.projection1 =  nn.Conv1d(in_channels=self.n_group[e_layers-1],
                                                 out_channels=enc_in,
                                                 kernel_size=1,
                                                 bias=True)
            self.projection2 = nn.Conv1d(in_channels=d_model,
                                                 out_channels=self.forecast_step,
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
            # self.projection = nn.Linear(d_model, c_out, bias=True)
            self.projection=Reconstruct(d_model,c_out,ngroup=self.n_group[-1],nsensor=nsensor[0],R_arch=self.R_arch)

    def forward(self, x):
        #batch_size,nsensor,window
        enc_out = self.embedding(x)
        #batch_size,nsensor,dmodel
        # print('enc_out',enc_out.size())
        enc_out,sensor_assoc = self.encoder(enc_out)
       
        # print('global enc_out',enc_out.size())
        if self.task=='Forecast':
           
            
            p1=torch.relu(self.projection1(enc_out))
            # print('p1',p1.size())
            p1=torch.transpose(p1,1,2)
            # print('pt',p1.size())
            enc_out=self.projection2(p1)
            # print('p2',enc_out.size())
            enc_out=torch.transpose(enc_out,1,2)
        elif self.task=='Encode':
            # print('global enc_out',enc_out.size())
            enc_out=rearrange(enc_out, 'b g l -> b (g l)')
            # print('enc_out',enc_out.size())
            enc_out=self.projection(enc_out)
        else:
            enc_out = self.projection(enc_out)
        #B,G,d_model
        # print('project enc_out',enc_out.size())
        if self.output_attention:
            return enc_out, sensor_assoc#, prior, sigmas
        else:
            return enc_out  # [B, L, D]
class DualADSensor(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 n_group=5,nsensor=51,forecast_step=1,device=torch.device('cuda:0'),task='C',
                 R_arch='LSTM',
                 revin=True,affine = True, subtract_last = False,output_encoding=False,sensor_embed='linear',
                  dropout=0.0, activation='gelu', output_attention=True,config={}):
        super(DualADSensor, self).__init__()
        #non-shared L2S
        self.win_size= win_size
        self.sensor_embed=sensor_embed
        self.c_out=c_out
        self.output_attention = output_attention
        self.n_group=n_group
        self.device=device
        self.task=task
        self.R_arch=R_arch
        self.forecast_step=forecast_step
        self.e_layers=e_layers
        self.d_model=d_model
        self.enc_in=enc_in
        self.nsensor=nsensor
        self.n_heads=n_heads
        self.revin=revin
        self.output_attention=output_attention
        self.output_encoding = output_encoding
        
        self.config=config
        
        
        self.local_model=ADSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
                        n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
                        nsensor=self.nsensor,forecast_step=self.forecast_step,
                        output_attention=self.output_attention,
                        device=self.device,task=self.task,
                        revin=self.revin,output_encoding=self.output_encoding,R_arch=self.R_arch,sensor_embed=self.sensor_embed,
                        args=self.config)
        # self.global_model=GlobalSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
        #                 n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
        #                 nsensor=self.nsensor,forecast_step=self.forecast_step,R_arch=self.R_arch,
        #                 output_attention=self.output_attention,
        #                 device=self.device,task=self.task,args=self.config)
        self.global_model=GlobalSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
                       n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
                       nsensor=self.nsensor,forecast_step=self.forecast_step,R_arch=self.R_arch,
                       output_attention=self.output_attention,
                       device=self.device,task=self.task,args=self.config)
    def forward(self,x):
        if self.output_attention:
            local_enc_out , local_assoc=self.local_model(x)
            global_enc_out , global_assoc = self.global_model(x)
            return local_enc_out,local_assoc,global_enc_out,global_assoc
        else:
            local_enc_out , local_assoc=self.local_model(x)
            global_enc_out , global_assoc = self.global_model(x)
            return local_enc_out,global_enc_out
# class DualADSensor(nn.Module):
#     def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
#                   dropout=0.0, activation='gelu', output_attention=True):
#         super(DualADSensor, self).__init__()
#         #non-shared L2S
#         self.local_model=ADSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
#                        n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
#                        nsensor=self.nsensor,forecast_step=self.forecast_step,
#                        device=self.device,task=self.task,
#                        revin=self.revin,output_encoding=self.output_encoding,R_arch=self.R_arch,
#                        args=self.config)
#         self.global_model=GlobalSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
#                        n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
#                        nsensor=self.nsensor,forecast_step=self.forecast_step,R_arch=self.R_arch,
#                        device=self.device,task=self.task,args=self.config)
#     def forward(self,x):
#         if self.output_attention:
#             local_enc_out , local_assoc=self.local_model(x)
#             global_enc_out , global_assoc = self.global_model(x)
#             return local_enc_out,local_assoc,global_enc_out,global_assoc
#         else:
#             local_enc_out , local_assoc=self.local_model(x)
#             global_enc_out , global_assoc = self.global_model(x)
#             return local_enc_out,global_enc_out