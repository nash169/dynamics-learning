#!/usr/bin/env python

import torch
import torch.nn as nn

class Spiral(nn.Module):
    def __init__(self):
        super(Spiral, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        W = torch.tensor([[-0.1, -2.0],
                         [2.0, -0.1]])
        self.lin.weight = nn.Parameter(W)
    def forward(self, t, x):
        return self.lin(x**3)