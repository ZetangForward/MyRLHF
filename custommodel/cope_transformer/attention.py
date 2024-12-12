import torch
from torch import nn
from einops import rearrange,repeat
import math
from typing import Literal

if __name__=="__main__":
    import sys,os
    sys.path.append(os.getcwd())


from model.backend.normalize import *



def get_mask(t,valid_len,zero_triu):
    '''
    t        : int or tuple  (q len,k len)
    valid_len: (b, ) (k len)
    '''
    qt,kt=(1,t) if isinstance(t,int) else t

    if valid_len is None:
        return None
    if valid_len.ndim>2:# means valid_len is actually a mask not len 
        return valid_len
    mask=(torch.arange(kt).to(valid_len.device)[None,:]<valid_len[:,None]) # (b, kt)
    mask=mask[:,None,:].repeat(1,qt,1) # (b, qt, kt)
    if zero_triu: # decoder self attention
        mask=torch.tril(mask)
    mask=mask[:,None,:,:] # (b, 1(h), qt, kt)
    return ~mask

    
class AbsPositionEncoding(nn.Module):
    def __init__(self,num_dims,dropout,max_len=6000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
        position=torch.arange(max_len,dtype=torch.float32)[:,None]
        inv_denominator=torch.exp(
            torch.arange(0,num_dims,2).float()\
            *-(4.*torch.log(torch.tensor(10.0))/num_dims)
        )

        self.sin=torch.sin(position*inv_denominator)
        self.cos=torch.cos(position*inv_denominator)
        

    def forward(self,x,current_steps=None):
        if self.sin.device!=x.device:
            self.sin=self.sin.to(x.device)
            self.cos=self.cos.to(x.device)
        x1,x2=x[..., 0::2], x[..., 1::2]
        if current_steps is None:
            sin,cos=self.sin[:x.size(2)][None,None,:,:],self.cos[:x.size(2)][None,None,:,:]
        else:
            sin,cos=self.sin[[current_steps]][None,None,:,:],self.cos[[current_steps]][None,None,:,:]
        return torch.stack([x1 * sin + x2 * (-cos), x1 * (-cos) + x2 * (-sin)], dim=-1).flatten(-2)

class ContextualPositionEmbedding(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max=npos_max
        self.pos_emb=nn.Parameter(torch.zeros(1, 1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        #query: (batch_size, num_heads, q_len, ndims)
        #attn : (batch_size, num_heads, q_len, k_len)
        gates=torch.sigmoid(attn_logits)
        pos=gates.flip(dim=-1).cumsum(dim=-1).flip(dim=-1)
        pos=pos.clamp(max=self.npos_max-1)  # (b, h, q_len, k_len)

        #interpolate from integar positions
        pos_ceil   = pos.ceil().long()
        pos_floor  = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb) #q(e^T) :(b, h, q_len, npos_max) 
        logits_ceil =logits_int.gather(dim=-1, index=pos_ceil)
        logits_floor=logits_int.gather(dim=-1, index=pos_floor)
        w = pos -pos_floor

        return logits_ceil * w + logits_floor * (1-w)
    

class SelfAttn(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.cope = ContextualPositionEmbedding(npos_max, head_dim)
        self.head_dim = head_dim

    def forward(self, query, key, val, mask):
        # q, k, v 's shape: (batch_size, seq_len, head_dim)
        attn_logits = torch.bmm(query, key.transpose(-1,-2))
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_logits += mask.log()
        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.bmm(attn, val)
        return out


class CopeMultiHeadAttention(nn.Module):
    def __init__(self,
                 ndims,
                 nheads,
                 npos_max,
                 dropout=0):
        super().__init__()
        assert ndims%nheads==0,"Input dims % num_heads != 0 !"
        self.ndims=ndims
        self.nheads=nheads
        self.per_head=ndims//nheads

        self.dropout=nn.Dropout(dropout)

        for i in ["q","k","v","o"]:
            setattr(self,"W"+i,nn.Linear(ndims,ndims))
            # nn.init.xavier_uniform_(getattr(self,"W"+i).weight)

        self.cope_enc=ContextualPositionEmbedding(npos_max=npos_max, head_dim=self.per_head)

    def forward_qkv(self,q,k,v):
        b,_,_=q.shape
        Q=self.Wq(q).view(b,q.size(1),-1,self.per_head).transpose(1,2)
        K=self.Wk(k).view(b,k.size(1),-1,self.per_head).transpose(1,2)
        V=self.Wv(v).view(b,v.size(1),-1,self.per_head).transpose(1,2)
        return Q,K,V
    
    def forward_attention(self,QK,V,mask=None):
        if type(QK) is tuple:
            Q,K=QK
            logits=torch.matmul(Q,K.transpose(-2,-1))/self.per_head**0.5
        else:logits=QK

        
        if mask is not None:
            neg_inf=torch.finfo(score.dtype).min
            score=score.masked_fill(mask,neg_inf)
            

        score=torch.softmax(logits,dim=-1)
        if mask is not None:
            score=score.masked_fill(mask,0.0)
        
        self.score=self.dropout(score)
        out=torch.matmul(self.score,V)
        out=self.Wo(out.transpose(1,2).reshape(V.size(0),-1,self.num_dims))
        
        return out

    def forward(self,q,k,v,mask=None,current_steps=None):
        '''
        valid_len shape: (bs,)
        '''
        b,_,_=q.shape
        
        Q,K,V=self.forward_qkv(q,k,v)
        
        PQ=self.abs_enc(Q,current_steps)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
        PK=self.abs_enc(K)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
        
        scores=self.get_scores(PQ,PK)
        
        return self.forward_attention(scores,V,mask)



class RoMultiHeadAttention(nn.Module):
    def __init__(self,num_dims,num_heads,dropout=0):
        super().__init__()
        assert num_dims%num_heads==0,"Input dims % num_heads != 0 !"
        self.num_dims=num_dims
        self.num_heads=num_heads
        self.per_head=num_dims//num_heads
        assert self.per_head%2==0,"RoPE needs the dim per head %2==0 !"

        self.dropout=nn.Dropout(dropout)

        for i in ["q","k","v","o"]:
            setattr(self,"W"+i,nn.Linear(num_dims,num_dims))
            # nn.init.xavier_uniform_(getattr(self,"W"+i).weight)

        self.abs_enc=AbsPositionEncoding(num_dims=self.per_head,dropout=dropout)

    def forward_qkv(self,q,k,v):
        b,_,_=q.shape
        Q=self.Wq(q).view(b,q.size(1),-1,self.per_head).transpose(1,2)
        K=self.Wk(k).view(b,k.size(1),-1,self.per_head).transpose(1,2)
        V=self.Wv(v).view(b,v.size(1),-1,self.per_head).transpose(1,2)
        return Q,K,V
    
    def forward_attention(self,QK,V,mask=None):
        if type(QK) is tuple:
            Q,K=QK
            score=torch.matmul(Q,K.transpose(-2,-1))/self.per_head**0.5
            
        else:score=QK

        
        if mask is not None:
            neg_inf=torch.finfo(score.dtype).min
            score=score.masked_fill(mask,neg_inf)
            

        
        # if mask is not None:
        #     assert mask.shape[-2:]==score.shape[-2:],\
        #         f"Mask:{mask.shape}, Score:{score.shape}"
        score=torch.softmax(score,dim=-1)
        if mask is not None:
            score=score.masked_fill(mask,0.0)
        
        self.score=self.dropout(score)
        out=torch.matmul(self.score,V)
        out=self.Wo(out.transpose(1,2).reshape(V.size(0),-1,self.num_dims))
        
        return out
    
    def get_scores(self,Q,K):
        return torch.matmul(Q,K.transpose(-2,-1))


    def forward(self,q,k,v,mask=None,current_steps=None):
        '''
        valid_len shape: (bs,)
        '''
        b,_,_=q.shape
        
        Q,K,V=self.forward_qkv(q,k,v)
        
        PQ=self.abs_enc(Q,current_steps)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
        PK=self.abs_enc(K)#.transpose(1,2).reshape(K.size(0),-1,self.num_dims)
        
        scores=self.get_scores(PQ,PK)
        
        return self.forward_attention(scores,V,mask)
