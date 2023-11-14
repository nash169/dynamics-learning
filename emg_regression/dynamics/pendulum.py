#!/usr/bin/env python

import torch
import torch.nn as nn

class Pendulum(nn.Module):
    def __init__(self, length):
        super(Pendulum, self).__init__()
        
        # params
        self._length = length
        self._gravity = 9.81
    
    def forward(self, t, x):
        y = torch.zeros_like(x)
        y[:,0] = x[:,1]
        y[:,1] = self.gravity/self.length*x[:,0].sin()
        if hasattr(self,'controller'):
            y[:,1] += self.controller(t,x)


    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value