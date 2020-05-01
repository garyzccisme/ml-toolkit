import torch
import torch.nn as nn
import torch.nn.functional as F


class Exp(nn.Module):
    def __init__(self):
        """
        This Module is designed as an exponential activation layer
        """
        super(Exp, self).__init__()

    def forward(self, input: torch.autograd.Variable) -> torch.autograd.Variable:
        return torch.exp(input)

