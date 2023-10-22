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
train_x = np.load('data/train_x.npy') # x: u_theta, u_phi
train_y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Select input and output
# Option 1: 
# input  = train_x # u1,u2
# output = train_y[:,:2] # theta, phi
# print("Input: u1(k), u2(k) ")
# print("Output: theta(k), phi(k)")

# Option 2:
input  = np.append(train_y[:-1,:2], train_x[:-1,:],axis=1) # theta(k), phi(k), u1(k), u2(k)
output = train_y[1:,:2] # theta(k+1), phi(k+1)
print("Input: theta(k), phi(k), u1(k), u2(k) ")
print("Output: theta(k+1), phi(k+1) ")

# Option 3: (works with any network)
# input  = np.append(train_y[:-1,:], train_x[:-1,:],axis=1) # theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k)
# output = train_y[1:,:2] # theta(k+1), phi(k+1)
# print("Input: theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k) ")
# print("Output: theta(k+1), phi(k+1) ")

# Normalize data
mu, std = input.mean(0), input.std(0)
input = (input - mu)/std

# Convert to torch
train_x = torch.from_numpy(input).float().to(device)
train_y = torch.from_numpy(output).float().to(device)
dim_input, dim_output  = train_x.shape[1], train_y.shape[1]
# print(dim_input,dim_output)

# approximator
# model = 'Feedforward'
model = 'LSTM'

# FeedForward NN
if model == 'Feedforward':
    num_neurons = [64]
    num_layers = 2
    approximator = FeedForward(dim_input, num_neurons*num_layers,dim_output).to(device)

# LSTM
if model == 'LSTM':
    num_layers = 2
    dim_hidden = 20  #32
    dim_pre_output = 20  #32
    bidirectional = False
    window_size = 5
    offset = 1
    approximator = LSTM(dim_input, dim_hidden, num_layers,
                        dim_pre_output, dim_output, bidirectional).to(device)
    train_x, train_y = approximator.process_input(train_x,train_y,window_size,offset)


# train
num_epochs = 300
optimizer = torch.optim.Adam(approximator.parameters(), lr=1e-2, weight_decay=1e-6)

loss_fun = torch.nn.MSELoss()
loss_log = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred = approximator(train_x)
    loss = loss_fun(pred,train_y)
    loss_log.append(loss.item())
    loss.backward()
    optimizer.step()

loss_log = np.array(loss_log)

fig, ax = plt.subplots()
ax.plot(np.arange(num_epochs), loss_log)
ax.set_title('MSE:'+str(np.round(loss.detach().numpy(),4)))
plt.show()
# TorchHelper.save(approximator, 'data/')

# Test on training set
ypred = pred.detach().numpy()
train_y = np.array(train_y)
train_loss = loss.detach().numpy()


fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(train_y[:,0],label='True')
ax.plot(ypred[:,0], label='Predicted')
ax.legend()
ax.set_title(f'Training data (MSE = {train_loss:.5f})')
ax = fig.add_subplot(212)
ax.plot(train_y[:,1], label='True')
ax.plot(ypred[:,1], label='Predicted')
ax.legend()
plt.show()


# Test data
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')

# Select input and output
input  = np.append(test_y[:-1,:2], test_x[:-1,:], axis=1) # theta(k), phi(k), u1(k), u2(k)
output = test_y[1:,:2]                                    # theta(k+1), phi(k+1)

# Normalize x
# input = (input - mu)/std
input = (input - input.mean(0))/input.std(0)

test_x = torch.from_numpy(input).float().to(device)
test_y = torch.from_numpy(output).float().to(device)

if model == 'LSTM':
    test_x, test_y = approximator.process_input(test_x,test_y,window_size,offset)

test_pred = approximator(test_x)
test_loss = loss_fun(test_pred,test_y)

test_pred = test_pred.detach().numpy()
test_y = np.array(test_y)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(test_y[:,0],label='True')
ax.plot(test_pred[:,0], label='Predicted')
ax.legend()
ax.set_title(f'Testing data (MSE = {test_loss:.5f})')
ax = fig.add_subplot(212)
ax.plot(test_y[:,1], label='True')
ax.plot(test_pred[:,1], label='Predicted')
ax.legend()
plt.show()