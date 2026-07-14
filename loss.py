import torch

def LogProb(logits, tao: float = 1.0):
    logits = (logits - torch.max(logits, dim = -1, keepdim = True).values)/tao
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    return log_probs


def CrossEntropy(inputs: torch.Tensor, targets: torch.Tensor):
    probs = LogProb(inputs)
    loss = probs.gather(dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    return -loss.mean()