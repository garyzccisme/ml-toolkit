import torch
import torch.nn as nn
from torch.distributions import Normal


class MultiClassLoss(nn.Module):
    def __init__(self, pre_log: bool = True):
        """
        This Module is designed as a customized loss criterion for multi-classification,
        implementing - mean(log(likelihood)).
        Args:
            pre_log: True if net_output has already been log transformed.
        """
        super().__init__()
        self.pre_log = pre_log

    def forward(self, net_output: torch.autograd.Variable, y: torch.Tensor) -> torch.autograd.Variable:
        """
        Args:
            net_output: Predicted values, size should be [n, num_class], n is the number of records.
            y: True values.
        Returns: loss function variable
        """
        if self.pre_log:
            log_likelihoods = net_output.gather(dim=1, index=y.type(torch.long).unsqueeze(1))
        else:
            log_likelihoods = net_output.log().gather(dim=1, index=y.type(torch.long).unsqueeze(1))
        return - log_likelihoods.mean()


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma: int = 0, pre_log: bool = True):
        """
        This Module is designed as a customized loss criterion for multi-classification,
        implementing - mean(log(likelihood)).
        Args:
            gamma: The parameter of focal factor.
            pre_log: True if net_output has already been log transformed.
        """
        super().__init__()
        self.gamma = gamma
        self.pre_log = pre_log

    def forward(self, net_output: torch.autograd.Variable, y: torch.Tensor) -> torch.autograd.Variable:
        """
        Args:
            net_output: Predicted values, size should be [n, num_class], n is the number of records.
            y: True values.
        Returns: loss function variable
        """
        if self.pre_log:
            log_likelihoods = net_output.gather(dim=1, index=y.type(torch.long).unsqueeze(1))
        else:
            log_likelihoods = net_output.log().gather(dim=1, index=y.type(torch.long).unsqueeze(1))
        return - (log_likelihoods * (1 - log_likelihoods.exp()) ** self.gamma).mean()


class NormalLoss(nn.Module):
    def __init__(self):
        """
        This Module is designed as a customized loss criterion for Normal distribution,
        implementing - mean(log(likelihood)).
        """
        super(NormalLoss, self).__init__()

    def forward(self, net_output: torch.autograd.Variable, y: torch.Tensor) -> torch.autograd.Variable:
        """
        Args:
            net_output: Predicted values, size should be [n, 2], n is the number of records.
            y: True values
        Returns: loss function variable
        """
        param_mu = net_output.t()[0]
        param_sigma = net_output.t()[1]
        predictive_dist = Normal(loc=param_mu, scale=param_sigma)
        log_likelihoods = predictive_dist.log_prob(y)
        return -log_likelihoods.sum()


