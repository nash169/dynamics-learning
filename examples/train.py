#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

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

data = np.load('data/'+ds_name+'.npy')
if ds_name == 'human':
    data = data[:,:params['train']['num_trajectories'],:]
    input_dim = data.shape[-1]
    output_dim =  params['dimension']*2

# data
data = data[:, :, :input_dim]
if params['train']['position'] and params['order'] == 'second':
    data = np.delete(data, np.arange(params['dimension'], 2*params['dimension']), axis=2)
    input_dim -= params['dimension']
    output_dim -= params['dimension']

if params['train']['padding']:  # this not make sense for autonomous systems
    pad_len = 3*params['window_size']
    x_pad = np.repeat(data[0][np.newaxis,:], pad_len, axis=0)
    x_pad_end = np.repeat(data[-1][np.newaxis,:], pad_len, axis=0)
    # x_pad[:,:,output_dim:] = np.zeros((x_pad.shape[0],x_pad.shape[1],input_dim-output_dim))
    data = np.append(x_pad, data, axis=0)
    # data = np.append(data,x_pad_end, axis=0)

train_x = torch.from_numpy(data).float().to(device)
train_x = train_x.unfold(0, params['window_size'], params['window_step']).permute(0, 1, 3, 2)[:-1].reshape(-1, params['window_size'], input_dim)

train_y = train_x[params['window_size']::params['window_step']]
train_y = torch.from_numpy(data[params['window_size']::params['window_step']]).float().to(device)
train_y = train_y.reshape(-1, input_dim)[:, :output_dim]

# model
if params['model']['net'] == 'rnn':
    model = RNN(input_size=input_dim, hidden_dim=params['model']['hidden_dim'], output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'lstm':
    model = LSTM(input_size=input_dim, 
                 hidden_dim=params['model']['hidden_dim'], 
                 pre_output_size=params['model']['preoutput_size'],
                 output_size=output_dim, n_layers=params['model']['num_layers']).to(device)
elif params['model']['net'] == 'node':
    model = NODE(input_size=input_dim, structure=[params['model']['hidden_dim']]*params['model']['num_layers'], output_size=output_dim, time_step=params['step_size']).to(device)
    train_x = train_x[:, -1, :]
else:
    print("Function approximator not supported.")
    sys.exit(0)
if params['train']['load']:
    TorchHelper.load(model, 'models/'+ds_name+'_'+params['model']['net'], device)
    model.train()

# normalize
if params['train']['normalize_input']:
    mu_u, std_u = train_x[:,:,output_dim:].mean(dim=(0,1)), train_x[:,:,output_dim:].std(dim=(0,1))
    train_x[:,:,output_dim:] = (train_x[:,:,output_dim:] - mu_u)/ std_u
    mu_BL = train_x[0,:,output_dim:]
    # train_x[:,:,output_dim:] = train_x[:,:,output_dim:] - mu_BL
    np.save('data/'+ds_name+'_mu_u', mu_u.cpu())
    np.save('data/'+ds_name+'_std_u',std_u.cpu())

if params['train']['normalize_output']:
    mu_y, std_y = train_x[:,:,:output_dim].mean(dim=(0,1)), train_x[:,:,:output_dim].std(dim=(0,1))      
    train_x[:,:,:output_dim] = (train_x[:,:,:output_dim] - mu_y)/ std_y
    train_y = (train_y - mu_y)/std_y
    np.save('data/'+ds_name+'_mu_y', mu_y.cpu())
    np.save('data/'+ds_name+'_std_y',std_y.cpu())

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
    loss = loss_fun(pred, train_y)
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

if params['dimension'] == 2:
    # plot to check dataset build
    fig, ax = plt.subplots()
    if params['model']['net'] == 'node':
        ax.scatter(train_x[0, 0], train_x[0, 1], c='r')
    else:
        ax.scatter(train_x[0, :, 0], train_x[0, :, 1], c='r')
    ax.scatter(train_y[0, 0], train_y[0, 1], c='g')
    ax.plot(data[:20, 0, 0], data[:20, 0, 1], c='b')
    fig.tight_layout()
    fig.savefig("media/"+ds_name+"_check.png", format="png", dpi=100, bbox_inches="tight")
    fig.clf()

if ds_name == 'human':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(data.shape[1]):
        ax.plot(data[:, i, 0], data[:, i, 1],zorder=1)
        ax.scatter(data[0, i, 0], data[0, i, 1], c='b', zorder=3)
        ax.scatter(data[-1, i, 0],data[-1, i, 1],c='r', zorder=3)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("media/"+ds_name+"_train.png", format="png", dpi=100, bbox_inches="tight")
    fig.clf()

plt.close('all')