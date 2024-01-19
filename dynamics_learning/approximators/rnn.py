#!/usr/bin/env python

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, 
                          hidden_dim, 
                          n_layers,
                          nonlinearity='relu',
                          batch_first=True) # RNN hidden units
        self.fc = nn.Linear(hidden_dim, output_size) # output layer
    def forward(self, x):
        bs, _, _ = x.shape
        h0 = torch.zeros(self.n_layers, bs,
                        self.hidden_dim).requires_grad_().to(x.device)
        out, hidden = self.rnn(x, h0.detach())
        out = out.view(bs, -1, self.hidden_dim)
        out = self.fc(out)
        return out[:, -1, :]