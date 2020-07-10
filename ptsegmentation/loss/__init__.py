import torch

from ptsegmentation.loss.CrossEntropy import cross_entropy
from ptsegmentation.loss.loss import cross_entropy2d


def get_loss_function(name):
    return {
        "cross_entropy": cross_entropy,
        "cross_entropy2d": cross_entropy2d,
        "mse": torch.nn.MSELoss()
    }[name]
