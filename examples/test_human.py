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
ds_name = 'human'
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
elif ds_name == 'human':
    pass
else:
    print("DS not supported.")
    sys.exit(0)


# initial state
if params['test']['train_data']:
    test_data = np.load('data/'+ds_name+'.npy')
    pad_len = 0 
    if params['test']['padding']:
        pad_len = 3*params['window_size']
        x_pad = np.repeat(test_data[0][np.newaxis,:], pad_len, axis=0)
        x_pad_end = np.repeat(test_data[-1][np.newaxis,:], pad_len, axis=0)
        test_data = np.append(x_pad, test_data, axis=0)
        # test_data = np.append(test_data,x_pad_end, axis=0)

    x0 = torch.from_numpy(test_data[0]).float().to(device)[:params['test']['num_trajectories']]
    if ds_name == 'human':
        x = torch.from_numpy(test_data).float().to(device)[:,:params['test']['num_trajectories'],:]
        x0 = x[0]
        input_dim = x0.shape[-1]
else:
    x0 = TorchHelper.grid_uniform(torch.tensor(params['test']['grid_center']),
                                  torch.tensor(params['test']['grid_size']),
                                  params['test']['num_trajectories']).to(device)
    if params['simulate']['fixed_state']:
        data = torch.from_numpy(np.load('data/'+ds_name+'.npy')[0][0]).float().to(device)
        x0.data[:, :input_dim] = data[:input_dim]

x0_net = x0[:, :input_dim]

if params['train']['position'] and params['order'] == 'second':
    input_dim -= params['dimension']
    output_dim -= params['dimension']    
    x = torch.cat((x[:,:,:output_dim], x[:,:,output_dim+params['dimension']:]), dim=2)
    x0_net = torch.cat((x0_net[:, :output_dim], x0_net[:, output_dim+params['dimension']:]), dim=1)

# normalize
if params['train']['normalize_input']:
    mu_u  = torch.from_numpy(np.load('data/'+ds_name+'_mu_u'+'.npy')).to(device)
    std_u = torch.from_numpy(np.load('data/'+ds_name+'_std_u'+'.npy')).to(device)
    x[:,:,output_dim:] = (x[:,:,output_dim:] - mu_u)/ std_u

if params['train']['normalize_output']:
    mu_y  = torch.from_numpy(np.load('data/'+ds_name+'_mu_y'+'.npy')).to(device)
    std_y = torch.from_numpy(np.load('data/'+ds_name+'_std_y'+'.npy')).to(device)
    x[:,:,:output_dim] = (x[:,:,:output_dim] - mu_y)/ std_y


# integration timeline
t = torch.arange(0.0, params['test']['duration'], params['step_size']).to(device)

# solution (time, trajectory, dimension)
if ds_name != 'human':
    with torch.no_grad():
        x = odeint(ds, x0, t, method='rk4')
    x = x.permute(1, 0, 2)
else:
    u = x[:,:,output_dim:]

if params['controlled']:
    u = torch.zeros(x.shape[0], x.shape[1], params['dimension'])
    u[:-1, :, :] = x[1:, :, 2*params['dimension']:3*params['dimension']]
    x[:, :, 2*params['dimension']:3*params['dimension']] = u
    

# model
if params['model']['net'] == 'rnn':
    model = RNN(input_size=input_dim, hidden_dim=params['model']['hidden_dim'], output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'lstm':
    model = LSTM(input_size=input_dim, hidden_dim=params['model']['hidden_dim'], pre_output_size=params['model']['preoutput_size'],output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'node':
    model = NODE(input_size=input_dim, structure=[params['model']['hidden_dim']]*params['model']['num_layers'], output_size=output_dim, time_step=params['step_size']).to(device)
else:
    print("Function approximator not supported.")
    sys.exit(0)
TorchHelper.load(model, 'models/'+ds_name+'_'+params['model']['net'], device)
model.eval()

# generate model trajectories
with torch.no_grad():
    if params['model']['net'] == 'node':
        x_net = model(x0_net, t).permute(1, 0, 2)
    else:
        x_net = x0_net.reshape(params['test']['num_trajectories'], 1, input_dim).repeat(1, params['window_size'], 1)
        # this solution is more realistic but then it is better to insert some sample like this one in the training set
        # x_net = x0_net.reshape(params['test']['num_trajectories'], 1, input_dim)
        # x_net = x.permute(1,0,2)[:, :params['window_size'],:].to(device)
        
        for i in range(0,pad_len + len(t)-1):
            
            y_net = model(x_net[:, -params['window_size']:, :])  # .reshape(params['test']['num_trajectories'], 1, output_dim)
            if params['controlled']:
                u_net = ctr(t[i], torch.cat((y_net, x0[:, y_net.shape[1]:]), dim=1))[:, params['dimension']:2*params['dimension']]
                # u_net = u[:, i, :].to(device)
            else:
                u_net = u[i+1,:, :]
            y_net = torch.cat((y_net, u_net), dim=1)
            x_net = torch.cat((x_net, y_net.reshape(params['test']['num_trajectories'], 1, input_dim)), dim=1)


# move data to cpu
x = x.cpu()
x_net = x_net.cpu()
t = t.cpu()

# plot
if params['dimension'] == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # colors = plt.cm.get_cmap('hsv', params['test']['num_trajectories'])
    for i in range(params['test']['num_trajectories']):
    # for i in range(1):
        # ax.scatter(x[i, 0, 0], x[i, 0, 1], c='k')
        # ax.plot(x[i, :, 0], x[i, :, 1], linewidth=2.0)  # color=colors(i)
        # ax.plot(x_net[i, :, 0], x_net[i, :, 1], linestyle='dashed', linewidth=2.0, color='k')
        ax.scatter(x[0,i, 0], x[0,i, 1], c='k')
        ax.plot(x[:,i,0], x[:,i,1], linewidth=2.0)  # color=colors(i)
        ax.plot(x_net[i,:,0], x_net[i,:,1], linestyle='dashed', linewidth=2.0, color='k')
    fig.tight_layout()
    fig.savefig("media/"+ds_name+'_'+params['model']['net']+"_test.png", format="png", dpi=100, bbox_inches="tight")
    fig.clf()
plt.close('all')

N = params['test']['num_trajectories']
fig, ax = plt.subplots(3, N, figsize=(20,6))  # Adjust figsize as needed
# fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust hspace and wspace as needed
labels = [r'$\theta$', r'$\phi$']
for traj in range(N):
    for i in range(len(labels)):
        ax[i,traj].plot(t, x[pad_len:, traj, i], label=labels[i])
        ax[i,traj].plot(t[1:], x_net[traj, pad_len+params['window_size']:, i], c='r', linestyle='--')
        ax[i,traj].set_xlabel('Time [s]')
        ax[i,0].set_ylabel(labels[i])
    ax[0,traj].set_title(f'Trajectory {traj+1}')
    i+=1
    ax[i,traj].plot(x[:, traj, 0], x[:, traj, 1])
    ax[i,traj].plot(x_net[traj, params['window_size']:,0],x_net[traj, params['window_size']:,1], c='r', linestyle='--')
    ax[i,traj].set_xlabel(r'$\theta$')
    ax[i,traj].set_ylabel(r'$\phi$')
fig.legend(labels=['True', 'Predicted'], loc='upper right')  # Add a common legend
plt.tight_layout()  # Adjust layout to prevent overlapping
fig.show()
