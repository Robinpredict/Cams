import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

from einops import rearrange
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
class GlobalAttention2(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, 
                 # nnodes=51,
                 embedding_dim=5,
                 n_group=5, nsensor=5,n_heads=1,
                 output_attention=False,device=torch.device('cuda:0'), Select_Att=None):
        super(GlobalAttention2, self).__init__()
        self.scale = scale
        self.gc=graph_constructor(nsensor, nsensor, embedding_dim, device).to(device)
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.k=k
        # if Select_Att is None:
        #     self.Select_Att=SelectAttention(win_size, n_group, nsensor, n_heads,device=device)
        # else:
        #     self.Select_Att=Select_Att
        self.W = nn.Parameter(torch.randn(n_heads,nsensor,n_group), requires_grad=True).to(device)
        # window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

    def forward(self,idx, queries, keys, values, attn_mask,x):
        # B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        # scale = self.scale or 1. / sqrt(E)
        # print(idx.is_cuda,x.is_cuda)
        scores = self.gc(idx)# N,N
        #B,H,N,Group
        # select_matrix=self.Select_Att(x)
        # select_matrix=select_matrix[:,0,:,:]
        #static selected matrix
        #
        #BNG
        # scores=self.mask_score(scores,self.k)
        #B,N,group
        scores=torch.einsum("ls,blg->bsg", scores, self.W)
        
        # attn = scale * scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        

        series = torch.softmax(scores, dim=-1)
        #series:N,N
        #values:B,N,H,dmodel
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        V =torch.einsum("bls,blg->bsg", x,series)
        #V:B,Length,NGroup
        if self.output_attention:
            return (V.contiguous(), series)
        else:
            return (V.contiguous(), None)
class SensorAnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, output_attention=False,
                 n_group=5, nsensor=5, n_heads=1,device=torch.device('cuda:0')):
        super(SensorAnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.k=k
        self.Select_Att=SelectAttention(win_size, n_group, nsensor, n_heads,device=device)
        self.GlobalAttention=GlobalAttention2(win_size,mask_flag=False, scale=None,k=3, attention_dropout=0.0, 
                     # nnodes=51,
                     embedding_dim=5,
                     n_group=5, nsensor=5,n_heads=1,
                     output_attention=False,device=device, Select_Att=None)
        # window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

    def forward(self, idx,queries, keys, values, attn_mask,x=None):
        #x batch_size,nsensors,seq_len
        raw_q=queries
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        #B,H,N,N
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # print('scores',scores.size())
       
        #B,H,N,Ngroup
        #select B,H,N,Ngroup#torch.Size([2, 1, 51, 5])
        select_matrix=self.Select_Att(x)
        # print('select',select_matrix.shape)
        scores=torch.einsum("bhls,bhlg->bhsg", scores, select_matrix)
        #B,H,N,Ngroup
        # print('s scores',scores.size())
        attn = scale * scores
        series = self.dropout(torch.softmax(attn, dim=-1))
        #
        V =torch.einsum("blhs,bhlg->bshg", values,series)
        # print('V',V.size())
        global_out, global_series = self.GlobalAttention(
            idx,
            queries,
            keys,
            values,
            attn_mask,
            x=raw_q#batch_size,nsensors,seq_len
        )
        if self.output_attention:
            return (V.contiguous(), series,global_out.contiguous(),global_series)
        else:
            return (V.contiguous(), None, global_out.contiguous(), None)
class SensorAnomalyAttentionLayer(nn.Module):
    def __init__(self, local_attention,global_attention, d_model, n_heads,n_group=5, d_keys=None,idx=0,
                  d_values=None):
        super(SensorAnomalyAttentionLayer, self).__init__()
        self.local_att_layer=AttentionLayer(local_attention, d_model, n_heads,n_group,d_keys,d_values)
        self.global_att_layer=GAttentionLayer(global_attention, d_model, n_heads,n_group, d_keys,idx,
                      d_values)
    def forward(self,queries, keys, values,global_queries, global_keys, global_values, attn_mask=None):
        local_out,local_series=self.local_att_layer(queries, keys, values, attn_mask)
        global_out,global_series=self.global_att_layer(global_queries, global_keys, global_values, attn_mask)
        return local_out,local_series,global_out,global_series

class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

class SelectAttention(nn.Module):
    def __init__(self,seq_len,n_group,nsensor,n_heads,d_sel=1,device=torch.device('cuda:0')):
        super(SelectAttention,self).__init__()
        self.n_group=n_group
        self.nsensor=nsensor
        self.n_heads=n_heads
        self.seq_len=seq_len
        self.d_sel=d_sel
        MLP=nn.ModuleList()
        MLP.append(nn.Linear(seq_len,n_heads*d_sel))
        MLP.append(nn.Tanh())
        self.MLP=nn.Sequential(*MLP)
        self.MLP.to(device)
        self.W = nn.Parameter(torch.randn(n_heads, n_group,d_sel), requires_grad=True).to(device)
        
    def forward(self,x):
        # print(x.shape)
        #batch_size,nsensors,seq_len
        B,N,L=x.shape
        x=rearrange(x, 'b n l-> (b n) l')#(queries, 'b l s -> b s l')
        newx=self.MLP(x)#(b*n,nhead*d_sel)
        newx=torch.reshape(newx,(B,self.n_heads,N,self.d_sel))
        #rearrange(ims, '(b1 b2) h w c -> b1 b2 h w c ', b1=2)
        # newx=rearrange(newx,'(b n) (h,d) -> b n h d',b=B,h=self.n_heads)
        # newx=rearrange(newx, 'b n h d-> b h n d')
        # B,H,Nsensor,Ngroup
        select_=torch.einsum("bhnd,hgd->bhng", newx, self.W)
        select_=torch.relu(select_)
        # select_[select_<0.1]=1
        return select_
        
class SensorAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, output_attention=False,
                 n_group=5, nsensor=5, n_heads=1,device=torch.device('cuda:0')):
        super(SensorAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.k=k
        self.Select_Att=SelectAttention(win_size, n_group, nsensor, n_heads,device=device)
        # window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, attn_mask,x=None):
        #x batch_size,nsensors,seq_len
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        #B,H,N,N
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # print('scores',scores.size())
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)
        # scores=self.mask_score(scores,self.k)
        #B,H,N,Ngroup
        #select B,H,N,Ngroup#torch.Size([2, 1, 51, 5])
        select_matrix=self.Select_Att(x)
        # print('select',select_matrix.shape)
        scores=torch.einsum("bhls,bhlg->bhsg", scores, select_matrix)
        #B,H,N,Ngroup
        # print('s scores',scores.size())
        attn = scale * scores
        

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        

        series = self.dropout(torch.softmax(attn, dim=-1))
        #series:B,H,N,N
        #values:B,N,H,dmodel
        
        # V = torch.einsum("bhls,bshd->blhd", series, values)
        #series:B,H,N,Ngroup
        # print(series.size())
        #values:B,N,H,L
        # print(values.size())
        V =torch.einsum("blhs,bhlg->bshg", values,series)
        # print('V',V.size())
        if self.output_attention:
            return (V.contiguous(), series)
        else:
            return (V.contiguous(), None)
    def mask_score(self,scores,k=2):
        #scores: B,H,N,N
        B,H,N,_=scores.size()
        # print(scores[b,h,:,n])
        # print(scores[-1,-1,:,-1])
        mask_=torch.zeros(B,H,N,N)
        for b in range(B):
            for h in range(H):
                for n in range(N):
                    sc=scores[b,h,:,n]
                    _v,idx=torch.topk(sc, k,dim=-1)
                    mask_[b,h,:,n][idx]=1
        # if K is None:
        # _v,idx=torch.topk(scores, k,dim=-1)
        # print(idx.size())
        # print(idx[0].size())
        # mask_=mask_[idx]
        # print(mask_[b,h,:,n])
        scores=scores*mask_
        # print(scores[b,h,:,n])
        return scores[:,:,:self.n_group,:] 
class GlobalAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, 
                 # nnodes=51,
                 embedding_dim=5,
                 n_group=5, nsensor=5,n_heads=1,
                 output_attention=False,device=torch.device('cuda:0'), Select_Att=None):
        super(GlobalAttention, self).__init__()
        self.scale = scale
        self.gc=graph_constructor(nsensor, nsensor, embedding_dim, device).to(device)
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.k=k
        if Select_Att is None:
            self.Select_Att=SelectAttention(win_size, n_group, nsensor, n_heads,device=device)
        else:
            self.Select_Att=Select_Att
        # window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

    def forward(self,idx, queries, keys, values, attn_mask,x):
        # B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        # scale = self.scale or 1. / sqrt(E)
        # print(idx.is_cuda,x.is_cuda)
        scores = self.gc(idx)# N,N
        #B,H,N,Group
        select_matrix=self.Select_Att(x)
        select_matrix=select_matrix[:,0,:,:]
        #static selected matrix
        #
        #BNG
        # scores=self.mask_score(scores,self.k)
        #B,N,group
        scores=torch.einsum("ls,blg->bsg", scores, select_matrix)
        
        # attn = scale * scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        

        series = torch.softmax(scores, dim=-1)
        #series:N,N
        #values:B,N,H,dmodel
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        V =torch.einsum("bls,blg->bsg", x,series)
        #V:B,Length,NGroup
        if self.output_attention:
            return (V.contiguous(), series)
        else:
            return (V.contiguous(), None)
    def mask_score(self,scores,k=2):
        #scores: B,H,N,N
        N,_=scores.size()
        # print(scores[b,h,:,n])
        # print(scores[-1,-1,:,-1])
        mask_=torch.zeros(N,N)
        # for b in range(B):
            # for h in range(H):
        for n in range(N):
            sc=scores[:,n]
            _v,idx=torch.topk(sc, k,dim=-1)
            mask_[:,n][idx]=1
        
        scores=scores*mask_
        # print(scores[b,h,:,n])
        return scores[:,:,:5,:] 

class GlobalAttention2_(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,k=3, attention_dropout=0.0, 
                 # nnodes=51,
                 embedding_dim=5,
                 n_group=5, nsensor=5,n_heads=1,
                 output_attention=False,device=torch.device('cuda:0'), Select_Att=None):
        super(GlobalAttention2_, self).__init__()
        self.scale = scale
        self.gc=graph_constructor(nsensor, nsensor, embedding_dim, device).to(device)
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.k=k
        if Select_Att is None:
            self.Select_Att=SelectAttention(win_size, n_group, nsensor, n_heads,device=device)
        else:
            self.Select_Att=Select_Att
        # window_size = win_size
        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

    def forward(self,idx, queries, keys, values, attn_mask,x):
        # B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        # scale = self.scale or 1. / sqrt(E)
        # print(idx.is_cuda,x.is_cuda)
        scores = self.gc(idx)# N,N
        #B,H,N,Group
        select_matrix=self.Select_Att(x)
        # select_matrix=select_matrix[:,0,:,:]
        #static selected matrix
        #
        #BNG
        # scores=self.mask_score(scores,self.k)
        #B,N,group
        scores=torch.einsum("ls,bhlg->bhsg", scores, select_matrix)
        
        # attn = scale * scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        

        series = torch.softmax(scores, dim=-1)
        #series:N,N
        #values:B,N,H,dmodel
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        V =torch.einsum("bls,blg->bsg", x,series)
        #V:B,Length,NGroup
        if self.output_attention:
            return (V.contiguous(), series)
        else:
            return (V.contiguous(), None)
    def mask_score(self,scores,k=2):
        #scores: B,H,N,N
        N,_=scores.size()
        # print(scores[b,h,:,n])
        # print(scores[-1,-1,:,-1])
        mask_=torch.zeros(N,N)
        # for b in range(B):
            # for h in range(H):
        for n in range(N):
            sc=scores[:,n]
            _v,idx=torch.topk(sc, k,dim=-1)
            mask_[:,n][idx]=1
        
        scores=scores*mask_
        # print(scores[b,h,:,n])
        return scores[:,:,:5,:] 

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        #num_nodes, subgraph_size, node_dim, device, alpha, static_feat
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)#num_embeddings, embedding_dim
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        #node idx
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
        # print(idx.shape)#nnode
        # print('before linear',nodevec1.shape)#nnode,node_dim#lim1 dim dim
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        # print('after linear',nodevec1.shape)#nnode,node_dim
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        #nnode,nnode
        # adj = F.relu(torch.tanh(self.alpha*a))
        adj=torch.tanh(self.alpha*a)
        #nnode,nnode
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        #zeros(nnode,nnode)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        #s1: value t1: index
        mask.scatter_(1,t1,s1.fill_(1))#assign 1 to those topk positions
        adj = adj*mask#multiply by adj
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,n_group=5, d_keys=None,
                  d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.n_group=n_group
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        #qk dimension is the same
        #kv length is the same
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        # self.sigma_projection = nn.Linear(d_model,
                                          # n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        #_: d_model
        #qkv:batch_size,nsensors,seq_len
        raw_q=queries
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        # print('q',queries.size())
        # print(B,L,H)
        # print(self.query_projection(queries).size())
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        #after projection
        #qkv:B,nsensor,H,dkey
        # print('values',values.size())
        # sigma = self.sigma_projection(x).view(B, L, H)
        #Q:batch_size,window,head,dquery
        #kv:batch_size,window,head,dkey
        #out: blhd
        #series: bshd
        #outs: batch,length,head,d_values
        out, series = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            x=raw_q#batch_size,nsensors,seq_len
        )
        #B,H,Nsensor,nsensor
        # print('sensor',series.size())
        #B,dv,H,Group
        # print(out.size())
        
        out = out.view(B, self.n_group, -1)
        
        return self.out_projection(out), series#, prior, sigma
class GAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,n_group=5, d_keys=None,idx=0,
                  d_values=None):
        super(GAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.n_group=n_group
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.idx=idx
        #qk dimension is the same
        #kv length is the same
        # self.query_projection = nn.Linear(d_model,
        #                                   d_keys * n_heads)
        # self.key_projection = nn.Linear(d_model,
        #                                 d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        # self.sigma_projection = nn.Linear(d_model,
                                          # n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        #_: d_model
        #qkv:batch_size,nsensors,seq_len
        raw_q=queries
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        # print('q',queries.size())
        # print(B,L,H)
        # print(self.query_projection(queries).size())
        # queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        #after projection
        #qkv:B,nsensor,H,dkey
        # print('values',values.size())
        # sigma = self.sigma_projection(x).view(B, L, H)
        #Q:batch_size,window,head,dquery
        #kv:batch_size,window,head,dkey
        #out: blhd
        #series: bshd
        #outs: batch,length,head,d_values
        out, series = self.inner_attention(
            self.idx,
            0,
            0,
            values,
            attn_mask,
            x=raw_q#batch_size,nsensors,seq_len
        )
        #B,H,Nsensor,nsensor
        # print('sensor',series.size())
        #B,dv,H,Group
        #B,dv,NGroup
        # print(out.size())
        
        out = out.view(B, self.n_group, -1)
        
        return self.out_projection(out), series
    
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#         self.norm = nn.LayerNorm(d_model)
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model,
#                                           d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model,
#                                         d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model,
#                                           d_values * n_heads)
#         # self.sigma_projection = nn.Linear(d_model,
#                                           # n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)

#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask):
#         # here,sensor-association, seq_l should be dmodel
#         #_: d_model
#         #qkv:batch_size,window,dmodel or nsensors
#         B, L, nsensors = queries.shape
#         _, S, _ = keys.shape
#         queries= rearrange(queries, 'b l s -> b s l')
#         keys= rearrange(keys, 'b l s -> b s l')
#         values= rearrange(values, 'b l s -> b s l')
#         H = self.n_heads
#         x = queries
#         print('q',queries.size())
#         print(B,L,H)
#         print(self.query_projection(queries).size())
#         queries = self.query_projection(queries).view(B, nsensors, H, -1)
#         keys = self.key_projection(keys).view(B, nsensors, H, -1)
#         values = self.value_projection(values).view(B, nsensors, H, -1)
#         # sigma = self.sigma_projection(x).view(B, L, H)
#         #Q:batch_size,nsensor,head,dquery
#         #kv:batch_size,nsensor,head,dkey
#         #out: blhd
#         #series: bshd
#         #outs: batch,length,head,d_values
#         out, sensor_assoc= self.inner_attention(
#             queries,
#             keys,
#             values,
#             # sigma,
#             attn_mask
#         )
#         #sensor_assoc:batch,head,sensor,sensor
#         # print('senosr',sensor_assoc.size())
#         # out = out.view(B, L, -1)
#         print(out.size())
#         out = out.view(B,  nsensors,-1)
#         print('out',out.size())
#         out=self.out_projection(out)
#         # out=rearrange(out, 'b s l -> b l s')
#         return out, sensor_assoc
