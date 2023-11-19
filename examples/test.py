#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from torchdiffeq import odeint

from emg_regression.dynamics import Spiral, Pendulum
from emg_regression.approximators import RNN, LSTM, NODE
from emg_regression.utils.torch_helper import TorchHelper

# set torch device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configuration
ds_name = 'pendulum'
with open("configs/"+ds_name+".yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)
if params['model']['net'] == 'node':
    params['window_step'] = 1

if params['order'] == 'first':
    input_dim = params['dimension']
elif params['order'] == 'second':
    input_dim = 2*params['dimension']
output_dim = input_dim
if params['controlled']:
    input_dim += params['dimension']

# dynamics
if ds_name == 'spiral':
    ds = Spiral().to(device)
elif ds_name == 'pendulum':
    ds = Pendulum().to(device)

    def ctr(t, x):
        # modify last part of the state to record control signal
        x[:, 4] = x[:, 6]*torch.sin(x[:, 8]*t)
        x[:, 5] = -ds.gravity/ds.length*x[:, 1].sin() + x[:, 7]*torch.sin(x[:, 9]*t)

        # control input
        y = torch.zeros_like(x)
        y[:, 2] = x[:, 4]
        y[:, 3] = x[:, 5]
        return y

    ds.controller = ctr
else:
    print("DS not supported.")
    sys.exit(0)

# initial state
if params['test']['train_data']:
    x0 = torch.from_numpy(np.load('data/'+ds_name+'.npy')[0]).float().to(device)[:2]
    params['test']['num_trajectories'] = x0.shape[0]
else:
    x0 = TorchHelper.grid_uniform(torch.tensor(params['test']['grid_center']),
                                  torch.tensor(params['test']['grid_size']),
                                  params['test']['num_trajectories']).to(device)
    if params['simulate']['fixed_state']:
        data = torch.from_numpy(np.load('data/'+ds_name+'.npy')[0][0]).float().to(device)
        x0.data[:, :input_dim] = data[:input_dim]

# integration timeline
t = torch.arange(0.0, params['test']['duration'], params['step_size']).to(device)

# solution (time, trajectory, dimension)
with torch.no_grad():
    x = odeint(ds, x0, t)
x = x.permute(1, 0, 2)

# model
if params['model']['net'] == 'rnn':
    model = RNN(input_size=input_dim, hidden_dim=params['model']['hidden_dim'], output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'lstm':
    model = LSTM(input_size=input_dim, hidden_dim=params['model']['hidden_dim'], output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'node':
    model = NODE(input_size=input_dim, structure=[params['model']['hidden_dim']]*params['model']['num_layers'], output_size=output_dim, time_step=params['step_size']).to(device)
else:
    print("Function approximator not supported.")
    sys.exit(0)
TorchHelper.load(model, 'models/'+ds_name+'_'+params['model']['net'], device)

# generate model trajectories
with torch.no_grad():
    if params['model']['net'] == 'node':
        x_net = model(x0[:, :input_dim], t).permute(1, 0, 2)
    else:
        x_net = x0[:, :input_dim].reshape(params['test']['num_trajectories'], 1, input_dim).repeat(1, params['window_size'], 1)
        # this solution is more realistic but then it is better to insert some sample like this one in the training set
        # x_net = x0[:, :input_dim].reshape(params['test']['num_trajectories'], 1, input_dim)
        # x_net = x[:, :params['window_size'], :]
        for i in range(len(t)-1):
            y_net = model(x_net[:, -params['window_size']:, :])  # .reshape(params['test']['num_trajectories'], 1, output_dim)
            if params['controlled']:
                y_net = torch.cat((y_net, x0[:, output_dim:]), dim=1)
                ctr(t[i+1], y_net)
            x_net = torch.cat((x_net, y_net[:, :input_dim].reshape(params['test']['num_trajectories'], 1, input_dim)), dim=1)

# move data to cpu
x = x.cpu()
x_net = x_net.cpu()

# plot
if params['dimension'] == 2:
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
