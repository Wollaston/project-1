"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# via Project 0
class LogisticRegression(nn.Module):
    """Logistic regression model"""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        logits = torch.sigmoid(self.linear(x))
        return logits


# Basic NN ref: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class MLP(nn.Module):
    """Multilayer Perceptron Model, although activations
    are not limited to classic perceptrons"""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


# Basic CNN ref: https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# CNN NLP ref: https://colab.research.google.com/drive/1b7aZamr065WPuLpq9C4RU6irB59gbX_K#scrollTo=ejGLw8TKViBY
class CNN(nn.Module):
    """Convolutional Neural Net Model"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        filter_sizes=[1, 1, 1],
        num_filters=[100, 100, 100],
    ) -> None:
        super().__init__()
        # Conv Network
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=num_filters[i],
                    kernel_size=filter_sizes[i],
                )
                for i in range(len(filter_sizes))
            ]
        )
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), output_size)

    def forward(self, input):
        # Tensors come in as shape (batch_size, embed_size)
        # We add a dummy dimension and permute the shape
        # To match the expected input size for conv1d
        input = torch.unsqueeze(input, 1)
        input = input.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(input)) for conv1d in self.conv1d_list]
        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        logits = self.fc(x_fc)
        return logits
