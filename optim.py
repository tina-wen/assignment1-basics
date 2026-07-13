import torch
import math
from typing import Optional
from collections.abc import Callable

class AdamW(torch.optim.Optimizer):
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

def get_lr(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t/T_w * lr_max
    if t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)    
    return lr_min

class CosineScheduler():
    def __init__(self,optimizer,lr_max,lr_min,T_w,T_c):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_w = T_w
        self.T_c = T_c

    def step(self,t):
        lr = get_lr(t,self.lr_max,self.lr_min,self.T_w,self.T_c)
        for group in self.optimizer.param_groups:
            group['lr'] = lr



















































