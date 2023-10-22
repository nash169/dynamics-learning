#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_pre_output_layer, dim_output, bidirectional):
        super(LSTM, self).__init__()

        self.dim_input = dim_input

        self.dim_output = dim_output

        self.layer_dim = num_layers

        self.hidden_dim = dim_recurrent

        self.lstm = nn.LSTM(input_size=dim_input, 
                            hidden_size=dim_recurrent,
                            num_layers=num_layers,
                            bidirectional=bidirectional, 
                            batch_first=True)
        
        self.D = (2 if bidirectional else 1)

        self.fc_o2y = nn.Linear(in_features=dim_recurrent*self.D, out_features=dim_pre_output_layer)

        self.fc_y2c = nn.Linear(in_features=dim_pre_output_layer, out_features=dim_output)

    def forward(self, X):

        batch_size = X.size(0)
        # Initialize hidden state vector and cell state
        h0 = torch.zeros(self.D*self.layer_dim, batch_size, self.hidden_dim).to(X.device)
        c0 = torch.zeros(self.D*self.layer_dim, batch_size, self.hidden_dim).to(X.device)

        output, _ = self.lstm(X, (h0.detach(), c0.detach()))
        output = self.fc_o2y(output[:, -1, :])
        output = F.relu(output)
        output = self.fc_y2c(output)

        return output
    
    
    def process_input(self,x,y,window_size,offset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X0 = x
        Y0 = y
        X = torch.empty(1, window_size, self.dim_input).float().to(device)
        Y = torch.empty(1, self.dim_output).float().to(device)
        for k in range(0, X0.size(0)-window_size+1, offset):
            x = X0[k:k+window_size, :].unsqueeze(0)
            y = Y0[k+window_size-1].view(1,-1)
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
        X, Y = X[1:].float(), Y[1:,:].float()
        return X, Y