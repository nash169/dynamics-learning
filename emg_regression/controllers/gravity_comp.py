#!/usr/bin/env python

import torch.nn as nn

class GravityCompensation(nn.Module):
    def __init__(self, model):
        super(GravityCompensation, self).__init__()

        self.model = model
    
    def forward(self, t, x):
        return 0.9 * self.model.gravity(x)
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value