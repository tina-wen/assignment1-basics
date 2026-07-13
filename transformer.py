
import torch
import torch.nn as nn
from torch.nn import Module

from collections.abc import Callable, Iterable
import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        # self.in_feature = in_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3.0 * std, b = 3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)
        # return x@self.weight.T

    def load_weight(self, weights: torch.Tensor):
        with torch.no_grad():
            self.weight.data = weights.clone()
        
class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.weight, a = -3, b = 3)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
    
    def load_weight(self, weights: torch.Tensor):
        with torch.no_grad():
            self.weight.data = weights.clone()

class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
    
    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.sum(x ** 2, dim = -1) / self.d_model + self.eps).unsqueeze(-1)
        res = (x / RMS) * self.gain
        return res.to(in_type)
    
    def load_weight(self, weights: torch.Tensor):
        with torch.no_grad():
            self.gain.data = weights.clone()

class Swiglu(Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        def _Silu(y: torch.Tensor):
            return y * torch.sigmoid(y)
        return self.w2(_Silu(self.w1(x)) * self.w3(x))
    
    def load_weight(self, weight1: torch.Tensor, weight2: torch.Tensor, weight3: torch.Tensor):
        with torch.no_grad():
            self.w1.load_weight(weight1)
            self.w2.load_weight(weight2)
            self.w3.load_weight(weight3)

class Rope(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, dtype = None, device = None):
        super().__init__()
        freqs = 1 / theta ** (torch.arange(0,d_k,2,dtype = dtype, device = device) / d_k)
        positions = torch.arange(max_seq_len, dtype = dtype, device = device)
        cos_cache = torch.cos(positions[:,None] * freqs[None,:])
        self.cos = torch.empty(max_seq_len, d_k, device = device)
        self.cos[:,0::2] = cos_cache
        self.cos[:,1::2] = cos_cache


        sin_cache = torch.sin(positions[:,None] * freqs[None,:])
        self.sin = torch.empty(max_seq_len, d_k, device = device)
        self.sin[:,0::2] = sin_cache
        self.sin[:,1::2] = sin_cache
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        new_cos, new_sin = self.cos[token_positions], self.sin[token_positions]
        x_even, x_odd = x[...,0::2], x[...,1::2]
        x_rotated_even = x_even * new_cos[...,0::2] - x_odd * new_sin[...,0::2]
        x_rotated_odd = x_odd * new_cos[...,0::2] + x_even * new_sin[...,0::2]

        res = torch.empty_like(x)
        res[...,0::2] = x_rotated_even
        res[...,1::2] = x_rotated_odd
        return res

        
def Softmax(output: torch.Tensor, dim: int):
    exp_tensor = torch.exp(output - torch.max(output, dim = dim, keepdim = True).values)
    return exp_tensor / torch.sum(exp_tensor, dim = dim, keepdim = True)

# scaled_dot_product_attention: 主要就是缩放，注意力分数要除以一个sqrt(d_k)
def SDPA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = K.shape[-1]
    attn = Q @ K.transpose(-1,-2)/(d_k ** 0.5)
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))
    return Softmax(attn,-1) @ V


class MultiheadSelfAttn(Module):
    def __init__(self, d_model: int, num_heads: int, dtype = None, device = None):
        super().__init__()
        d_k, d_v = d_model // num_heads, d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.device = device

        self.q_proj = Linear(d_model, num_heads * d_k, device = device, dtype = dtype)
        self.k_proj = Linear(d_model, num_heads * d_k, device = device, dtype = dtype)
        self.v_proj = Linear(d_model, num_heads * d_v, device = device, dtype = dtype)
        self.o_proj = Linear(num_heads * d_v, d_model, device = device, dtype = dtype)
    

    def forward(self, x: torch.Tensor, max_seq_len: int | None = None, theta: float | None = None, token_positions: torch.Tensor | None = None):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(*query.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        key = key.view(*key.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        value = value.view(*value.shape[:-1], self.num_heads, self.d_v).transpose(-2,-3)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device = self.device)).bool()
        
        if theta is not None:
            my_rope = Rope(theta, self.d_k, max_seq_len, device = self.device)
            query = my_rope(query, token_positions)
            key = my_rope(key, token_positions)

        attn_val = SDPA(query, key, value, mask).transpose(-2,-3).contiguous()

        return self.o_proj(attn_val.view(*attn_val.shape[:-2],self.num_heads * self.d_v))
    
    def load_weight(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor, o_weight: torch.Tensor):
        with torch.no_grad():
            self.q_proj.weight.data = q_weight.clone()
            self.k_proj.weight.data = k_weight.clone()
            self.v_proj.weight.data = v_weight.clone()
            self.o_proj.weight.data = o_weight.clone()
    
class Block(Module):
    def __init__(self, d_model, num_heads, d_ff, device = None):
        super().__init__()
        self.num_heads = num_heads

        self.rms_norm1 = RMSNorm(d_model, device = device)
        self.mhsa = MultiheadSelfAttn(d_model,num_heads, device = device)
        self.rms_norm2 = RMSNorm(d_model, device = device)
        self.pwff = Swiglu(d_model,d_ff, device = device)

    def forward(self, x: torch.Tensor, max_seq_len: int, theta: float):
        batch,seq_len = x.shape[0],x.shape[1]
        token_positions = torch.arange(seq_len, device = x.device)
        x = x + self.mhsa(self.rms_norm1(x), max_seq_len, theta, token_positions)
        outputs = x + self.pwff(self.rms_norm2(x))
        return outputs

class TransLM(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int, device = None):
        super().__init__()
        self.layers = nn.ModuleList([Block(d_model, num_heads, d_ff, device) for _ in range(num_layers)])
        self.embedding = Embedding(vocab_size, d_model, device = device)
        self.ln_final = RMSNorm(d_model, device = device)
        self.out_embed = Linear(d_model, vocab_size, device = device)

    def forward(self, x: torch.Tensor, context_len: int, rope_theta: float):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, context_len, rope_theta)
        x = self.ln_final(x)
        x = self.out_embed(x)
        return x
