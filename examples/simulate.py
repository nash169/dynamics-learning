#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from torchdiffeq import odeint

from emg_regression.dynamics import Spiral, SphericalPendulum
from emg_regression.approximators.rnn import RNN
from emg_regression.utils.torch_helper import TorchHelper

# set torch device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configuration
ds_name = 'spiral'
with open("configs/"+ds_name+".yaml", "r") as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.SafeLoader)

# dynamics
if ds_name == 'spiral':
    ds = Spiral().to(device)
elif ds_name == 'spherical_pendulum':
    ds = SphericalPendulum().to(device)
else:
    print("DS not supported.")
    sys.exit(0)

# initial state
x0 = TorchHelper.grid_uniform(params['simulate']['grid_center'],
                              params['simulate']['grid_length'][0],
                              params['simulate']['grid_length'][1],
                              params['simulate']['num_trajectories']).to(device)

# integration timeline
t = torch.arange(0.0, params['simulate']['duration'], params['step_size']).to(device)

# solution (time, trajectory, dimension)
with torch.no_grad():
    x = odeint(ds, x0, t).cpu()
np.save('data/'+ds_name, x)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(params['simulate']['num_trajectories']):
    ax.scatter(x[0, i, 0], x[0, i, 1], c='k')
    ax.plot(x[:, i, 0], x[:, i, 1])
fig.tight_layout()
fig.savefig("media/"+ds_name+"_train.png", format="png", dpi=100, bbox_inches="tight")
fig.clf()
