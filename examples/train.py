#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt

from emg_regression.models.pendulum import Pendulum
from emg_regression.approximators.feedforward import FeedForward

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# data
train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')
train_x = torch.from_numpy(train_x[:,np.newaxis]).float().to(device)
train_y = torch.from_numpy(train_y[:,np.newaxis]).float().to(device)

# model
model = Pendulum(length=2.0)

# approximator
num_neurons = [64]
num_layers = 2
approximator = FeedForward(model.input_dim, num_neurons*num_layers, model.dim).to(device)

# train
num_epochs = 100
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
plt.show()
# TorchHelper.save(approximator, 'data/')
