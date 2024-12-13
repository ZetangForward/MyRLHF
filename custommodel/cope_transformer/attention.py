import torch
from torch import nn

from normalize import *



class ContextualPositionEmbedding(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max=npos_max
        self.pos_emb=nn.Parameter(torch.zeros(1, 1, head_dim, npos_max))

    def forward(self, query, attn_logits, mask):
        #query: (batch_size, num_heads, q_len / 1, ndims)
        #attn_logits : (batch_size, num_heads, q_len / 1, k_len)
        #mask:  (b, 1, q_len, k_len) , None means evaluating and using kv cache
        if mask is not None:
            attn_logits += mask.log()
        gates=torch.sigmoid(attn_logits)
        pos=gates.flip(dim=-1).cumsum(dim=-1).flip(dim=-1)
        pos=pos.clamp(max=self.npos_max-1)  # (b, h, q_len, k_len)

        #interpolate from integar positions
        pos_ceil   = pos.ceil().long()
        pos_floor  = pos.floor().long() # (b, h, q_len, k_len)
        logits_int = torch.matmul(query, self.pos_emb) #q(e^T) :(b, h, q_len, npos_max) 
        logits_ceil =logits_int.gather(dim=-1, index=pos_ceil)
        logits_floor=logits_int.gather(dim=-1, index=pos_floor)
        w = pos - pos_floor

        attn_logits += logits_ceil * w + logits_floor * (1-w)

        return attn_logits
    
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
            logits = torch.matmul(Q,K.transpose(-2,-1))/self.per_head**0.5            
        else: logits = QK
        
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(mask, neg_inf)

        score = torch.softmax(logits, dim=-1)
        if mask is not None:
            score = score.masked_fill(mask, 0.0)
        
        self.score = self.dropout(score)
        out = torch.matmul(self.score, V)
        out = self.Wo(out.transpose(1,2).reshape(V.size(0), -1, self.num_dims))
        
        return out

    def forward(self,q, k, v, mask = None):
        Q, K, V = self.forward_qkv(q, k, v)
        logits = torch.matmul(Q,K.transpose(-2,-1))/self.per_head**0.5
        logits = self.cope_enc(Q, logits, mask)
        
        return self.forward_attention(logits, V, mask)
