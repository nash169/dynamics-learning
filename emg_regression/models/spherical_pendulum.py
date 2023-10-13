#!/usr/bin/env python

import torch
import numpy as np

class SphericalPendulum():
    def __init__(self, length):
        super(SphericalPendulum, self).__init__()

        # State dimension
        self.dim = 1

        # input dimension
        self.input_dim = 1

        # gravity
        self.gravity = 9.81

    def __call__(self, state, input = None):
        return 

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value