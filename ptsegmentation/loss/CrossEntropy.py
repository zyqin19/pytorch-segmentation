import torch
import torch.nn.functional as F

def cross_entropy(input, target, weight=None, size_average=True):

    target = target.squeeze(1).long()

    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss