import torch
import math

def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    maxs = torch.max(inputs, 1)[0]
    inputs = inputs - maxs.unsqueeze(1)


    loss = -inputs.gather(1, targets.unsqueeze(1))

    inputs = math.e ** inputs
    log_probs = torch.log(inputs.sum(dim=1, keepdim=True))
    loss = loss + log_probs
    return loss.mean()

def perplexity(inputs: torch.FloatTensor, targets: torch.LongTensor):
    ce_loss = cross_entropy(inputs, targets)
    return torch.exp(ce_loss)