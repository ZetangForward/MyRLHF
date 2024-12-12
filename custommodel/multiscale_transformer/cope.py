import math
import torch
from torch import nn

class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max=npos_max
        self.pos_emb=nn.Parameter(torch.zeros(1,head_dim,npos_max))

    def forward(self, query, attn_logits):
        #compute positions
        gates=torch.sigmoid(attn_logits)
        pos=gates.flip(dim=-1).cumsum(dim=-1).flip(dim=-1)
        pos=pos.clamp(max=self.npos_max-1)

        #interpolate from integar positions
        pos_ceil   = pos.ceil().long()
        pos_floor  = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil =logits_int.gather(dim=-1, index=pos_ceil)
        logits_floor=logits_int.gather(dim=-1, index=pos_floor)
        w = pos -pos_floor

        return logits_ceil * w + logits_floor * (1-w)
    

class SelfAttn(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.cope = CoPE(npos_max, head_dim)
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
