#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import get_input_output, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data_x = np.load('data/train_x.npy') # x: u_theta, u_phi
data_y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Option 1: input: u1,u2, output: theta, phi
# Option 2: input: theta(k), phi(k), u1(k), u2(k), output: theta(k+1), phi(k+1)
train_x, train_y = get_input_output(data_x,data_y,option=1)
train_y = (train_y - train_y.mean(0))/train_y.std(0)
dt = 0.01             # 100 Hz , 100 samples/ second
t = np.arange(0,len(train_x),dt)

# FeedForward NN
num_neurons = [64]
num_layers = 2

# LSTM
dim_input  = train_x.shape[1]
dim_output = train_y.shape[1]
num_layers = 2
dim_hidden = 20
dim_pre_output = 20
bidirectional = False
window_size = 15
offset = 8

# train
num_epochs = 200
learning_rate = 1e-3
weight_decay  = 1e-5
training_ratio = 1
mini_batch_size = 10

# Model
# Feedforward
# approximator = FeedForward(dim_input, num_neurons*num_layers,dim_output).to(device)

#LSTM
approximator = LSTM(dim_input, dim_hidden, num_layers,
                    dim_pre_output, dim_output, bidirectional)

XTrain, YTrain, t_train = approximator.process_input(train_x, window_size, offset,
                                                    y=train_y, time=t)
# print(XTrain.shape, YTrain.shape, t_train.shape)


# Train
trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize=True,
                epochs=num_epochs,
                batch=mini_batch_size,
                shuffle=False,
                record_loss=True)

trainer.optimizer = torch.optim.Adam(approximator.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)

loss = trainer.train()
trained_model = trainer.model
plt.plot(trainer.losses)
plt.show()

mse_train, _ = evaluate_model(trained_model,XTrain,YTrain,vis=1)

# Test
# Testing data
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')

test_x, test_y = get_input_output(test_x,test_y,option=1)
# test_x = (test_x - train_x.mean(0))/train_x.std(0)
# test_y = (test_y - test_y.mean(0))/test_y.std(0)

XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
                                                  y=test_y, time=t)

mse_test, _ = evaluate_model(trained_model,XTest,YTest,t_train=t_test,vis=1)