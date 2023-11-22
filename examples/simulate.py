#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from torchdiffeq import odeint

from emg_regression.dynamics import Spiral, Pendulum
from emg_regression.utils.torch_helper import TorchHelper

# set torch device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configuration
ds_name = 'pendulum'
with open("configs/"+ds_name+".yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)

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
x0 = TorchHelper.grid_uniform(torch.tensor(params['simulate']['grid_center']),
                              torch.tensor(params['simulate']['grid_size']),
                              params['simulate']['num_trajectories']).to(device)
if params['simulate']['fixed_state']:
    pos = torch.clone(x0.data[0, :params['dimension']])
    x0.data[:, :params['dimension']] = pos

# integration timeline
t = torch.arange(0.0, params['simulate']['duration'], params['step_size']).to(device)

# solution (time, trajectory, dimension)
with torch.no_grad():
    x = odeint(ds, x0, t, method='rk4').cpu()

if params['order'] == 'first':
    input_dim = params['dimension']
elif params['order'] == 'second':
    input_dim = 2*params['dimension']

if params['controlled']:
    u = torch.zeros(x.shape[0], x.shape[1], params['dimension'])
    u[:-1, :, :] = x[1:, :, input_dim:input_dim+params['dimension']]
    x[:, :, input_dim:input_dim+params['dimension']] = u

np.save('data/'+ds_name, x)

# plot
if params['dimension'] == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(params['simulate']['num_trajectories']):
        ax.scatter(x[0, i, 0], x[0, i, 1], c='k')
        ax.plot(x[:, i, 0], x[:, i, 1])
        ax.scatter(x[-1, i, 0], x[-1, i, 1], c='r')
    fig.tight_layout()
    fig.savefig("media/"+ds_name+"_train.png", format="png", dpi=100, bbox_inches="tight")
    fig.clf()
