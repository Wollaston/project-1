"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


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
class CNN(nn.Module):
    """Convolutional Neural Net Model"""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 6, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool1d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool1d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output
