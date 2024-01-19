#!/usr/bin/env python

import torch
import torch.nn as nn
from torchdiffeq import odeint


class NODE(nn.Module):
    def __init__(self, input_size, structure, output_size, time_step=0.01):
        super(NODE, self).__init__()

        structure = [input_size] + structure

        layers = nn.ModuleList()

        for i, _ in enumerate(structure[:-1]):
            layers.append(nn.Linear(structure[i], structure[i+1]))
            layers.append(nn.ReLU())  # Tanh

        layers.append(nn.Linear(structure[-1], output_size))

        self._net = nn.Sequential(*(layers[i] for i in range(len(layers))))

        self.delta = time_step

    def dynamics(self, t, x):
        return self._net(x)

    def forward(self, x, t=None):
        if t is None:
            return odeint(self.dynamics, x, torch.tensor([0.0, self.delta]).to(x.device), method='rk4')[-1]
        else:
            return odeint(self.dynamics, x, t, method='rk4')

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        self._delta = value
