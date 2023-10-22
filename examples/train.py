#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt

from emg_regression.models.spherical_pendulum import SphericalPendulum
from emg_regression.approximators.feedforward import FeedForward
from emg_regression.approximators.lstm import LSTM

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Train data
data_x = np.load('data/train_x.npy') # x: u_theta, u_phi
data_y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Test data
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')

# Select input and output
def get_input_output(train_x,train_y,option):
    if option == 1:
        input  = train_x # u1,u2
        output = train_y[:,:2] # theta, phi
        print("Input: u1(k), u2(k) -> Output: theta(k), phi(k)")

    if option == 2:
        input  = np.append(train_y[:-1,:2], train_x[:-1,:],axis=1) # theta(k), phi(k), u1(k), u2(k)
        output = train_y[1:,:2] # theta(k+1), phi(k+1)
        print("Input: theta(k), phi(k), u1(k), u2(k) -> Output: theta(k+1), phi(k+1) ")

    if option == 3: # Option 3: (works with any network)
        input  = np.append(train_y[:-1,:], train_x[:-1,:],axis=1) # theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k)
        output = train_y[1:,:2] # theta(k+1), phi(k+1)
        print("Input: theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k) -> Output: theta(k+1), phi(k+1)")
    return input, output

# approximator
# model = 'Feedforward'
model = 'LSTM'

# FeedForward NN
num_neurons = [64]
num_layers = 2

# LSTM
num_layers = 2
dim_hidden = 20  #32
dim_pre_output = 20  #32
bidirectional = False
window_size = 8
offset = 3

# train
num_epochs = 500
learning_rate = 1e-2
weight_decay  = 1e-4

loss_fun = torch.nn.MSELoss()
loss_log = []

# Create a figure with subplots for each input/output option
options = [1,2,3]
fig, axes = plt.subplots(3, len(options), figsize=(8,5))

for i, option in enumerate(options):
    input, output = get_input_output(data_x,data_y,option=option)

    # Normalize data
    mu, std = input.mean(0), input.std(0)
    input = (input - mu) / std

    # Convert to torch
    train_x = torch.from_numpy(input).float().to(device)
    train_y = torch.from_numpy(output).float().to(device)
    dim_input, dim_output = train_x.shape[1], train_y.shape[1]

    # Model
    if model == 'Feedforward':
        approximator = FeedForward(dim_input, num_neurons*num_layers,dim_output).to(device)

    if model == 'LSTM':
        approximator = LSTM(dim_input, dim_hidden, num_layers,
                            dim_pre_output, dim_output, bidirectional).to(device)
        train_x, train_y = approximator.process_input(train_x,train_y,window_size,offset)

    # Trainer
    optimizer = torch.optim.Adam(approximator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_log = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = approximator(train_x)
        loss = loss_fun(pred,train_y)
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()

    loss_log = np.array(loss_log)

    # Test on training set
    ypred_train = approximator(train_x)
    ypred = pred.detach().numpy()
    train_y = np.array(train_y)
    train_loss = loss.item()

    # MSE loss
    ax = axes[0,i] if len(options) > 1 else axes[0]
    ax.plot(np.arange(num_epochs), loss_log)
    ax.set_title('Option '+str(i+1)+ f' (MSE = {train_loss:.4f})')

    ax = axes[1,i] if len(options) > 1 else axes[1]
    ax.plot(train_y[:-1,0],label='True')
    ax.plot(ypred[1:,0], color='r', label='Predicted')

    ax = axes[2,i] if len(options) > 1 else axes[2]
    ax.plot(train_y[:-1,1],label='True')
    ax.plot(ypred[1:,1], color='r' ,label='Predicted')
    # ax.legend()

# Adjust the layout and display the figure
plt.tight_layout()
plt.show()

# Test data
test_x, test_y = get_input_output(test_x,test_y,option=option)

# Normalize data
test_x = (test_x - mu) / std
test_x = torch.from_numpy(test_x).float().to(device)
test_y = torch.from_numpy(test_y).float().to(device)
test_x, test_y = approximator.process_input(test_x,test_y,window_size,offset)

ypred_test = approximator(test_x)
test_loss = loss_fun(ypred_test,test_y).item()
ypred = ypred_test.detach().numpy()
ytest = np.array(test_y)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(ytest[:-1,0],label='True')
ax.plot(ypred[1:,0], color='r',label='Predicted')
ax.legend()
ax.set_title(f'Testing data (MSE = {test_loss:.5f})')
ax = fig.add_subplot(212)
ax.plot(ytest[:,1], label='True')
ax.plot(ypred[:,1], color='r',label='Predicted')
ax.legend()
plt.show()
