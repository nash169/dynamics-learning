#!/usr/bin/env python

import torch
import torch.nn as nn


class Pendulum(nn.Module):
    def __init__(self, length=1.0):
        super(Pendulum, self).__init__()

        # params
        self._length = length
        self.gravity = 9.81

    def forward(self, t, x):  # theta x[0], phi x[1]
        y = torch.zeros_like(x)
        y[:, :2] = x[:, 2:4]
        y[:, 2] = -2*x[:, 2]*x[:, 3]/(x[:, 1].tan()+1e-3)
        y[:, 3] = (x[:, 2].square()*x[:, 1].cos() + self.gravity/self.length)*x[:, 1].sin()  # elevation

        if hasattr(self, 'controller'):
            y += self.controller(t, x)

        return y

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
