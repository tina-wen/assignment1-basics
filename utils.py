import torch
import os
import typing


weight_mapping = {
        # 隐藏层内部的权重
        "attn.q_proj.weight":"mhsa.q_proj.weight",
        "attn.k_proj.weight":"mhsa.k_proj.weight",
        "attn.v_proj.weight":"mhsa.v_proj.weight",
        "attn.output_proj.weight":"mhsa.o_proj.weight",
        "ln1.weight":"rms_norm1.gain",
        "ffn.w1.weight":"pwff.w1.weight",
        "ffn.w2.weight":"pwff.w2.weight",
        "ffn.w3.weight":"pwff.w3.weight",
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

def GradClip(params, max_l2_norm, eps = 1e-06):
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
        total_grad_norm *= clip_k

    return total_grad_norm

def get_batch(x, batch_size: int, context_len: int, device: str):
    num_samples = len(x) - context_len
    import numpy as np
    indices = np.random.randint(0,num_samples,batch_size)
    offsets = np.arange(context_len+1)
    block_idx = indices[:,None] + offsets
    datasets = torch.from_numpy(x[block_idx].astype(np.int64)).to(device)
    return datasets[:,:-1], datasets[:,1:]

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    res = {
        'model_states':model.state_dict(),
        'optimizer_states':optimizer.state_dict(),
        'step':iteration,
        }
    torch.save(res,out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    res = torch.load(src)
    model.load_state_dict(res['model_states'])
    optimizer.load_state_dict(res['optimizer_states'])
    return res['step']

