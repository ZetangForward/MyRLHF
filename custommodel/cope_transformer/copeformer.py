from typing import Union
import torch
from torch import nn
from attention import get_mask, CopeMultiHeadAttention
from feedforward import *

# from template.e2e.utils import State, EvalModule


class CopeFormerLayer(nn.Module):
    def __init__(self, 
                 idims, 
                 hdims, 
                 nheads, 
                 npos_max, 
                 dropout=0., 
                 norm: Union[PreNorm, AddNorm] = AddNorm):
        super().__init__()
        self.attn=norm(
            idims,
            CopeMultiHeadAttention(idims, nheads, npos_max, dropout),
            dropout
        )

        self.ffn=PosWiseFFN(idims, hdims, dropout, norm)
    def forward(self, x, mask, cache = None):
        if cache is None:# while training
            x = self.attn(x, x, x, mask)
            x = self.ffn(x)
        else:# while evaluating
            q = x[:, [-1], :]
            q = self.attn(q, x, x, None)
            q = self.ffn(q)
            x = torch.concat([cache, q], dim = 1)
        return x

class CopeFormer(nn.Module, 
                #  EvalModule
                 ):
    def __init__(self, 
                 nlayers,
                 idims,
                 hdims,
                 nheads,
                 npos_max,
                 vocab_size = None,
                 dropout=0.,
                 norm: Union[PreNorm, AddNorm] = AddNorm):
        super().__init__()
        self.nlayers = nlayers
        self.idims = idims
        self.norm = norm
        self.tok_enc = nn.Embedding(vocab_size, idims)

        self.layers = nn.Sequential(*[
            CopeFormerLayer(idims, hdims, nheads, npos_max, dropout, norm) \
            for _ in range(nlayers)
        ])
        
        self.fc = nn.Linear(idims, vocab_size)
    def forward(self, x, x_valid_len = None):
        bs, seq_len, _ = x.shape
        x = self.tok_enc(x)
        if self.norm is AddNorm:
            x = self.norm(x)
        mask = get_mask((seq_len, seq_len), 
                        (seq_len* torch.ones(bs, dtype=torch.int64,device = x.device)) \
                            if x_valid_len is None else x_valid_len,
                        zero_triu = True)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        if self.norm is PreNorm:
            x=self.norm(x)
        
        out=self.fc(x)
        
        return out    

    
#     def initState(self, eState: State, *args, **kwargs):
#         batch_size=len(eState)
#         self.caches=[torch.zeros((batch_size,0,self.idims),
#                                  dtype=eState.dtype,
#                                  device=eState.device)\
#                     for _ in range(self.num_layers)]
        
#         return eState
    

#     def selectNext(self, bk_pref, bk_pred):
#         '''
#         select next cache after topk

#         bk_pref:  (batch_beam_size, )  the  last   ids
#         bk_pred:  (batch_beam_size, )  the current pinyin ids

#         choose prefix from self.caches
        
#         '''
#         for i in range(len(self.caches)):
#             self.caches[i]=State(self.caches[i]).setitem(bk_pref).feats


#     def scoring(self,eState: State, dState: State):
#         '''
#         while evaluating:
        
#         x_valid_len only need one ,because all dState has the same length
#         '''
#         x, x_valid_len = dState
#         x=self.tok_enc(x)

#         if self.norm is AddNorm:
#             x=self.norm(x)

#         new_caches=[]

#         for i,layer in enumerate(self.layers):
#             x=layer(x, mask = None, cache = self.caches[i])
#             new_caches.append(x)
        
#         self.caches=new_caches

#         if self.norm is PreNorm:
#             x=self.norm(x)
#         out=x[:,-1,:]
# #         if self.norm is PreNorm:
# #             out=self.norm(out)

#         return self.fc(out)

    
