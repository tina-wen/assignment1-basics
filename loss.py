import torch

def CrossEntropy(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs - torch.max(inputs, dim = -1, keepdim = True).values
    probs = inputs - torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True))
    loss = probs.gather(dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    return -loss.mean()