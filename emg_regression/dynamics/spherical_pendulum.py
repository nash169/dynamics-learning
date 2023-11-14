#!/usr/bin/env python

import torch
import torch.nn as nn

class SphericalPendulum(nn.Module):
    def __init__(self, length):
        super(SphericalPendulum, self).__init__()

        # params
        self._length = length
        self.gravity = 9.81
    
    def forward(self, t, x): # theta x[0], phi x[1]
        y = torch.zeros_like(x)
        y[:,:2] = x[:,-2:]
        y[:,2] = -2*x[:,2]*x[:,3]/(x[:,1].tan()+1e-3)
        y[:,3] = x[:,2].square()*x[:,1].sin()*x[:,1].cos() + self.gravity/self.length*x[:,1].sin()

        if hasattr(self,'controller'):
            y[:,2:] += self.controller(t,x)

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