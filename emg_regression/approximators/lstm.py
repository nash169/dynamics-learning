#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_dim, pre_output_size, output_size, n_layers):
        super(LSTM, self).__init__()

        # internal params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pre_output_size = pre_output_size 

        # network
        self.lstm = nn.LSTM(input_size, 
                            hidden_dim, 
                            n_layers,
                            batch_first=True, 
                            dropout=0.0, 
                            proj_size=0) # LSTM hidden units
        
        if self.pre_output_size == 0:
            self.fc = nn.Linear(hidden_dim, output_size) # output layer
        else:
            self.fc = nn.Linear(hidden_dim, pre_output_size) # output layer
            self.fc_2 = nn.Linear(pre_output_size, output_size) # output layer
        
    def forward(self, x):
        # batch size
        bs = x.shape[0]
        # hidden state
        h0 = torch.zeros(self.n_layers, bs, self.hidden_dim).to(x.device)
        # cell state
        c0 = torch.zeros_like(h0)

        out, (hidden, cell) = self.lstm(x, (h0, c0))
        out = out.view(bs, -1, self.hidden_dim)
        out = self.fc(out[:, -1, :])
        
        if self.pre_output_size != 0:
            out = F.relu(out)
            out = self.fc_2(out)

        return out