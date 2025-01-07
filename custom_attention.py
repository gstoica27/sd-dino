import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional


class Attention(nn.Module):
    # copied directly from openclip backend. Because nn.MultiheadAttention is really annoying
    def __init__(
            self,
            module: torch.nn.Module,
    ):
        super().__init__()
        if hasattr(module, 'in_proj_bias'):
            self.bias = True
        
        out_dim, in_dim = module.in_proj_weight.shape
        hidden_dim = out_dim // 3
        self.query = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        self.key = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        self.value = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        
        # replace weights
        self.query.weight.data = module.in_proj_weight[:hidden_dim, :]
        self.key.weight.data = module.in_proj_weight[hidden_dim:2*hidden_dim, :]
        self.value.weight.data = module.in_proj_weight[2*hidden_dim:, :]
        if self.bias:
            self.query.bias.data = module.in_proj_bias[:hidden_dim]
            self.key.bias.data = module.in_proj_bias[hidden_dim:2*hidden_dim]
            self.value.bias.data = module.in_proj_bias[2*hidden_dim:]
        
        self.batch_first = module.batch_first
        self.num_heads = module.num_heads
        self.attn_drop = 0.0

    def transpose_batch_tokens(self, x):
        x = x.transpose(0, 1)
    
    def forward(
        self, q_x, k_x: Optional[torch.Tensor]=None, v_x: Optional[torch.Tensor]=None, 
        attn_mask: Optional[torch.Tensor] = None
    ):
        # x = deepcopy(q_x) # [B,T,C]
        if k_x is None:
            k_x = deepcopy(q_x)
            v_x = deepcopy(q_x)
        
        if not self.batch_first:
            for elem in [q_x, k_x, v_x]:
                self.transpose_batch_tokens(elem)

        q = self.query(q_x)
        k = self.key(k_x)
        v = self.value(v_x)
        
        q = q.transpose(0, 1) # [T,B,C]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        T, B, C = q.shape
        q = q.reshape(T, B * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(T, B * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(T, B * self.num_heads, -1).transpose(0, 1) # [BxnH,T,C/nH]

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            pdb.set_trace()
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        assert attn.shape[-1] == T
        # attn = self.attn_drop(attn)
        x = torch.bmm(attn, v)
        x = x.transpose(1, 0).reshape(T, B, self.num_heads, -1).flatten(2)
        # x = x.reshape(T, B, C)

        if self.batch_first:
            x = x.transpose(0, 1)
        return x


class MergeHeads(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.out_proj = module.out_proj
        self.out_drop = nn.Dropout(module.dropout)
    
    def forward(self, x):
        x = self.out_proj(x)
        x = self.out_drop(x)
        return (x, )
    
    
class AttentionBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.attention = Attention(module)
        self.merge_heads = MergeHeads(module)
    
    def forward(
        self, q_x, k_x: Optional[torch.Tensor]=None, v_x: Optional[torch.Tensor]=None, 
        attn_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        return self.merge_heads(self.attention(q_x, k_x, v_x, attn_mask))