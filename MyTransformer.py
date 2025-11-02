
import torch
import torch.nn as nn
from torch.nn import Module
from typing import Optional
from collections.abc import Callable, Iterable
import math

class MyLinear(Module):
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
        
class MyEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.weight, a = -3, b = 3)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
    
    def load_weight(self, weights: torch.Tensor):
        with torch.no_grad():
            self.weight.data = weights.clone()

class MyRMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(d_model, device = device, dtype = dtype))
    
    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.sum(x ** 2, dim = -1) / self.d_model + self.eps).unsqueeze(-1)
        res = (x / RMS) * self.gain
        return res.to(in_type)
    
    def load_weight(self, weights: torch.Tensor):
        with torch.no_grad():
            self.gain.data = weights.clone()

class MySwiglu(Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(d_ff, d_model, device = device, dtype = dtype))
        self.weight2 = nn.Parameter(torch.empty(d_model, d_ff, device = device, dtype = dtype))
        self.weight3 = nn.Parameter(torch.empty(d_ff, d_model, device = device, dtype = dtype))

    def forward(self, x: torch.Tensor):
        def _Silu(y: torch.Tensor):
            return y * torch.sigmoid(y)
        return (_Silu(x @ self.weight1.T) * (x @ self.weight3.T)) @ self.weight2.T
    
    def load_weight(self, weight1: torch.Tensor, weight2: torch.Tensor, weight3: torch.Tensor):
        with torch.no_grad():
            self.weight1.data = weight1.clone()
            self.weight2.data = weight2.clone()
            self.weight3.data = weight3.clone()

class MyRope(Module):
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

        
def MySoftmax(output: torch.Tensor, dim: int):
    exp_tensor = torch.exp(output - torch.max(output, dim = dim, keepdim = True).values)
    return exp_tensor / torch.sum(exp_tensor, dim = dim, keepdim = True)

def MySDPA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = K.shape[-1]
    attn = Q @ K.transpose(-1,-2)/(d_k ** 0.5)
    if mask is not None:
        mask = torch.where(mask, torch.tensor(1.0), torch.tensor(float('-inf')))
        attn = attn * mask
        attn = torch.where(attn == float('inf'), torch.tensor(float('-inf')), attn)
    return MySoftmax(attn,-1) @ V


class MyMultiheadSelfAttn(Module):
    def __init__(self, d_model: int, num_heads: int, dtype = None, device = None):
        super().__init__()
        d_k, d_v = d_model // num_heads, d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_weight = nn.Parameter(torch.empty(num_heads * d_k, d_model, device = device, dtype = dtype))
        self.k_weight = nn.Parameter(torch.empty(num_heads * d_k, d_model, device = device, dtype = dtype))
        self.v_weight = nn.Parameter(torch.empty(num_heads * d_v, d_model, device = device, dtype = dtype))
        self.o_weight = nn.Parameter(torch.empty(d_model, num_heads * d_v, device = device, dtype = dtype))
    

    def forward(self, x: torch.Tensor, max_seq_len: int | None = None, theta: float | None = None, token_positions: torch.Tensor | None = None):
        query = x @ self.q_weight.T
        key = x @ self.k_weight.T
        value = x @ self.v_weight.T

        query = query.view(*query.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        key = key.view(*key.shape[:-1], self.num_heads, self.d_k).transpose(-2,-3)
        value = value.view(*value.shape[:-1], self.num_heads, self.d_v).transpose(-2,-3)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        if theta is not None:
            my_rope = MyRope(theta, self.d_k, max_seq_len)
            query = my_rope(query, token_positions)
            key = my_rope(key, token_positions)

        attn_val = MySDPA(query, key, value, mask).transpose(-2,-3).contiguous()

        return attn_val.view(*attn_val.shape[:-2],self.num_heads * self.d_v) @ self.o_weight.T
    
    def load_weight(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor, o_weight: torch.Tensor):
        with torch.no_grad():
            self.q_weight.data = q_weight.clone()
            self.k_weight.data = k_weight.clone()
            self.v_weight.data = v_weight.clone()
            self.o_weight.data = o_weight.clone()
    
class MyBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.num_heads = num_heads

        self.rms_norm1 = MyRMSNorm(d_model)
        self.mhsa = MyMultiheadSelfAttn(d_model,num_heads)
        self.rms_norm2 = MyRMSNorm(d_model)
        self.pwff = MySwiglu(d_model,d_ff)

    def forward(self, x: torch.Tensor, max_seq_len: int, theta: float):
        batch,seq_len = x.shape[0],x.shape[1]
        token_positions = torch.arange(seq_len)
        x = x + self.mhsa(self.rms_norm1(x), max_seq_len, theta, token_positions)
        outputs = x + self.pwff(self.rms_norm2(x))
        return outputs

weight_mapping = {
        # 隐藏层内部的权重
        "attn.q_proj.weight":"mhsa.q_weight",
        "attn.k_proj.weight":"mhsa.k_weight",
        "attn.v_proj.weight":"mhsa.v_weight",
        "attn.output_proj.weight":"mhsa.o_weight",
        "ln1.weight":"rms_norm1.gain",
        "ffn.w1.weight":"pwff.weight1",
        "ffn.w2.weight":"pwff.weight2",
        "ffn.w3.weight":"pwff.weight3",
        "ln2.weight":"rms_norm2.gain",
        # 隐藏层外部的权重
        "token_embeddings.weight":"embedding.weight",
        "ln_final.weight":"ln_final.gain",
        "lm_head.weight":"out_embed.weight",
    }

def load_weights_with_map(model, weights: dict[str, torch.Tensor]):
    state_dict = {}
    for ex_module, ex_tensor in weights.items():
        state_dict[weight_mapping[ex_module]] = ex_tensor
    model.load_state_dict(state_dict,strict = True)

class MyTransLM(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([MyBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.embedding = MyEmbedding(vocab_size, d_model)
        self.ln_final = MyRMSNorm(d_model)
        self.out_embed = MyLinear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, context_len: int, rope_theta: float):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, context_len, rope_theta)
        x = self.ln_final(x)
        x = self.out_embed(x)
        return x

def load_multi_weights(model, weights: dict[str, torch.Tensor]):
    new_state_dict = {}
    for key, val in weights.items():
        # 对于隐藏层内部的权重
        if key.startswith("layers."):
            parts = key.split('.')
            layer_no = parts[1]
            rest_name = ".".join(parts[2:])
            new_state_dict[f"layers.{layer_no}.{weight_mapping[rest_name]}"] = val
        else:
            new_state_dict[weight_mapping[key]] = val
    model.load_state_dict(new_state_dict,strict=True)

def MyCrossEntropy(inputs: torch.Tensor, targets: torch.Tensor):
    batch_size = inputs.shape[0]
    inputs = inputs - torch.max(inputs, dim = -1, keepdim = True).values
    probs = inputs - torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True))
    loss = torch.mean(probs[torch.arange(batch_size),targets])
    return -loss


class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {'lr': lr, 'weight_decay': weight_decay, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, eps, lamda = group['lr'], group['betas'][0], group['betas'][1], group['eps'], group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                m, v, t = state.get("m",torch.zeros_like(p.data)), state.get("v",torch.zeros_like(p.data)), state.get("t",1)

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                
                alpha_t = lr * math.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps) 
                p.data -= lr * lamda * p.data

                # 状态更新
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss


def MyScheduler(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t/T_w * lr_max
    if t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)    
    return lr_min

def MyGradClip(params, max_l2_norm, eps = 1e-06):
    total_grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_grad_norm += torch.sum(p.grad.data ** 2)
    
    total_grad_norm = total_grad_norm ** 0.5

    if total_grad_norm >= max_l2_norm:
        clip_k = max_l2_norm / (total_grad_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data *= clip_k

    return
            