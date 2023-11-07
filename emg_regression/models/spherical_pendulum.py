#!/usr/bin/env python

import torch
import numpy as np

class SphericalPendulum():
    def __init__(self, length):
        super(SphericalPendulum, self).__init__()

        # length
        self._length = length
        
        # State dimension
        self.dim = 2

        # input dimension
        self.input_dim = 2

        # gravity
        self.gravity = 9.81

        self.epsilon = 1e-3

    def __call__(self, state, input = None):
        u = np.array([0.0,0.0]) if input is None else input
        x1 = state[0] if len(state.shape) == 1 else state[:,0] # theta
        x2 = state[1] if len(state.shape) == 1 else state[:,1] # phi
        v1 = state[2] if len(state.shape) == 1 else state[:,2]
        v2 = state[3] if len(state.shape) == 1 else state[:,3]

        a1 = -2 * v1 * v2 * np.cos(x2)/np.sin(x2) + u[0] if np.abs(np.sin(x2)) > 0.5 else u[0] 
        a2 = v1**2 * np.sin(x2) * np.cos(x2) + (self.gravity/self.length) * np.sin(x2) + u[1]

        # print(a1,a2)
        # eps = 1.0 if abs(np.sin(x2)) <= 0.5 else 0.0
        # theta_ddot = -2 * theta_dot * phi_dot * np.cos(phi)/(np.sin(phi)+eps) + u_theta
        # phi_ddot   = theta_dot**2 * np.sin(phi) * np.cos(phi) + (self.gravity/self.length) * np.sin(phi) + u_phi

        return np.array([a1,a2])

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value