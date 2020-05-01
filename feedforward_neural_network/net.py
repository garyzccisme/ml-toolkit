from typing import List

import torch
import torch.nn as nn


class MultiClassNet(nn.Module):
    def __init__(self, input_channel: int, output_channel: int,
                 channel_list: List[int] = None, dropout_p: float = 0.1):
        """
        This Module is designed as a customized network for softmax distribution,
        all full connected layers and activation layer can be redesigned.

        Args:
            input_channel: the dimension number of design matrix, aka number of signals.
            output_channel: the dimension number of predicted values, aka number of values we want to predict.
            channel_list: customize the intermediate channels for the first two layers, should have length=2,
            because there are totally three layers in network.
            dropout_p: dropout ratio for Dropout layer
        """
        super(MultiClassNet, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        if channel_list:
            self.channel_list = channel_list
        else:
            # initialize default channel_list
            self.channel_list = [input_channel * 2, output_channel * 2]
        self.fc1 = torch.nn.Linear(input_channel, self.channel_list[0])
        self.activation1 = nn.ReLU()
        self.fc2 = torch.nn.Linear(self.channel_list[0], self.channel_list[1])
        self.activation2 = nn.ReLU()
        self.fc3 = torch.nn.Linear(self.channel_list[1], output_channel)
        self.activation3 = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.autograd.Variable) -> torch.autograd.Variable:
        """
        Let input x go though the network in a customized sequence,
        all layers can be reordered, make sure self.activation3 is the last layer

        Args:
            x: input value
        Returns: network output
        """
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation3(x)
        return x

