import torch
import torch.nn as nn
import torch.nn.functional as F

class deeplabv1(nn.Module):
    def __init__(self, n_classes=21):
        super(deeplabv1, self).__init__()
