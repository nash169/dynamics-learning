#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from emg_regression.approximators import RNN, LSTM
from emg_regression.utils.torch_helper import TorchHelper

# set torch device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configuration
ds_name = 'spiral'
with open("configs/"+ds_name+".yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)

# data
data = np.load('data/'+ds_name+'.npy')
train_x = torch.from_numpy(data).float().to(device)
train_x = train_x.unfold(0,params['window_size'],params['window_step']).permute(0,1,3,2)[:-1].reshape(-1,params['window_size'],params['dimension'])
train_y = torch.from_numpy(data[params['window_size']::params['window_step']]).float().to(device)
train_y = train_y.reshape(-1,params['dimension'])

# model
if params['model']['net'] == 'rnn':
    model = RNN(input_size=params['dimension'], hidden_dim=params['model']['hidden_dim'], output_size=params['dimension'], n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'lstm':
    model = LSTM(input_size=params['dimension'], hidden_dim=params['model']['hidden_dim'], output_size=params['dimension'], n_layers=params['model']['num_layers']).to(device)
else:
    print("Function approximator not supported.")
    sys.exit(0)

# train
optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'], weight_decay=params['train']['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, 
                                                       threshold=1e-3, threshold_mode='rel', cooldown=0, 
                                                       min_lr=0, eps=1e-8, verbose=True)
loss_fun = torch.nn.MSELoss()
loss_log = np.zeros(params['train']['num_epochs'])

for epoch in range(params['train']['num_epochs']):
    optimizer.zero_grad()
    pred = model(train_x)
    loss = loss_fun(pred,train_y)
    loss.backward()
    optimizer.step()
    if params['train']['dynamic_lr']:
      scheduler.step(loss)

    loss_log[epoch] = loss.item()
    if params['train']['verbose']:
       print("Epoch ", epoch, ": ", loss.item())

# save
TorchHelper.save(model, 'models/'+ds_name+'_'+params['model']['net'])

# move data to cpu
train_x = train_x.cpu()
train_y = train_y.cpu()

# plot loss
fig, ax = plt.subplots()
ax.plot(np.arange(params['train']['num_epochs']), loss_log)
fig.tight_layout()
fig.savefig("media/"+ds_name+'_'+params['model']['net']+"_loss.png", format="png", dpi=100, bbox_inches="tight")
fig.clf()

# plot to check dataset build
fig, ax = plt.subplots()
ax.scatter(train_x[0,:,0], train_x[0,:,1], c='r')
ax.scatter(train_y[0,0], train_y[0,1], c='g')
ax.plot(data[:20,0,0], data[:20,0,1], c='b')
fig.tight_layout()
fig.savefig("media/"+ds_name+"_check.png", format="png", dpi=100, bbox_inches="tight")
fig.clf()