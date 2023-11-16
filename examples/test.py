#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from torchdiffeq import odeint

from emg_regression.dynamics import Spiral, SphericalPendulum
from emg_regression.approximators import RNN, LSTM, NODE
from emg_regression.utils.torch_helper import TorchHelper

# set torch device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configuration
ds_name = 'spiral'
with open("configs/"+ds_name+".yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)
if params['model']['net'] == 'node':
    params['window_step'] = 1

# dynamics
if ds_name == 'spiral':
    ds = Spiral().to(device)
elif ds_name == 'spherical_pendulum':
    ds = SphericalPendulum().to(device)
else:
    print("DS not supported.")
    sys.exit(0)

# initial state
x0 = TorchHelper.grid_uniform(params['test']['grid_center'],
                              params['test']['grid_length'][0],
                              params['test']['grid_length'][1],
                              params['test']['num_trajectories']).to(device)

# integration timeline
t = torch.arange(0.0, params['test']['duration'], params['step_size']).to(device)

# solution (time, trajectory, dimension)
with torch.no_grad():
    x = odeint(ds, x0, t)
x = x.permute(1, 0, 2)

# model
if params['model']['net'] == 'rnn':
    model = RNN(input_size=params['dimension'], hidden_dim=params['model']['hidden_dim'], output_size=params['dimension'], n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'lstm':
    model = LSTM(input_size=params['dimension'], hidden_dim=params['model']['hidden_dim'], output_size=params['dimension'], n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'node':
    model = NODE(input_size=params['dimension'], structure=[params['model']['hidden_dim']]*params['model']['num_layers'], output_size=params['dimension'], time_step=params['step_size']).to(device)
else:
    print("Function approximator not supported.")
    sys.exit(0)
TorchHelper.load(model, 'models/'+ds_name+'_'+params['model']['net'], device)

# generate model trajectories
with torch.no_grad():
    if params['model']['net'] == 'node':
        x_net = model(x0, t).permute(1, 0, 2)
    else:
        x_net = x0.reshape(params['test']['num_trajectories'], 1, params['dimension']).repeat(1, params['window_size'], 1)
        # this solution is more realistic but thens it is better to insert some sample like this one in the training set
        # x_net = x0.reshape(params['test']['num_trajectories'], 1, params['dimension'])
        # x_net = x[:, :params['window_size'], :]
        with torch.no_grad():
            for _ in range(len(t)-1):
                x_net = torch.cat((x_net, model(x_net[:, -params['window_size']:, :]).reshape(params['test']['num_trajectories'], 1, params['dimension'])), dim=1)

# move data to cpu
x = x.cpu()
x_net = x_net.cpu()

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
# colors = plt.cm.get_cmap('hsv', params['test']['num_trajectories'])
for i in range(params['test']['num_trajectories']):
    ax.scatter(x[i, 0, 0], x[i, 0, 1], c='k')
    ax.plot(x[i, :, 0], x[i, :, 1], linewidth=2.0)  # color=colors(i)
    ax.plot(x_net[i, :, 0], x_net[i, :, 1], linestyle='dashed', linewidth=2.0, color='k')
fig.tight_layout()
fig.savefig("media/"+ds_name+'_'+params['model']['net']+"_test.png", format="png", dpi=100, bbox_inches="tight")
fig.clf()
