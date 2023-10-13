#!/usr/bin/env python

import torch
import numpy as np

class Integrator():
    def __init__(self, step):
        super(Integrator, self).__init__()

        self._step = step

    def __call__(self, dynamics, state, input):
        x = state[:dynamics.dim] if len(state.shape) == 1 else state[:,:dynamics.dim]
        v = state[dynamics.dim:] if len(state.shape) == 1 else state[:,dynamics.dim:]

        v = v + self.step*dynamics(state, input)
        x = x + self.step*v

        return np.concatenate((x,v))

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
