#!/usr/bin/env python

import torch
import numpy as np

class Pendulum():
    def __init__(self, length):
        super(Pendulum, self).__init__()
        # length
        self._length = length
        
        # State dimension
        self.dim = 1

        # input dimension
        self.input_dim = 1

        # gravity
        self.gravity = 9.81

    def __call__(self, state, input = None):
        u = 0 if input is None else input
        x = state[:self.dim] if len(state.shape) == 1 else state[:,:self.dim]
        # v = state[self.dim:] if len(x.shape) == 1 else state[:,self.dim:]

        a = self.gravity/self.length * np.sin(x) + u

        # theta_dd = g/l * sin(theta) + u
        return a

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value