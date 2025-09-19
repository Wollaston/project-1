"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression model"""
    raise NotImplementedError


class MLP(nn.Module):
    """Multilayer perceptron"""
    raise NotImplementedError


class CNN(nn.Module):
    """CNN model"""
    raise NotImplementedError