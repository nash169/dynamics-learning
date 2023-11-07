#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data = pickle.load(open('data/traj_center_100_3s.pkl','rb')) # x: u_theta, u_phi
t      = data['t']
x_traj = data['x'] # state vec: theta, phi, theta_dot, phi_dot
u_traj = data['u'] # u = uc + ug
uc_traj = data['uc']
ug_traj = data['ug']
nb_traj = len(x_traj)

# Dataset
# xin  = np.concatenate((u_traj[:,:-1,:], x_traj[:,:-1,:2]), axis=2)
# yout = x_traj[:,1:,:2]

# System identification
# xin  = u_traj[:,:,:]
# yout = x_traj[:,:,:2]

# Set training params
dt = 0.01             # 100 Hz , 100 samples/ second
fs = 1/dt
t_ = t[1:]

normalize_input  = False
normalize_output = False

# LSTM
dim_hidden = 40
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 0.1 #s
offset = 1

# train
mini_batch_size = 20
learning_rate = 1e-3
weight_decay  = 1e-6
training_ratio = 0.75
nb_epochs = 50

# Prepare data to make forward predictions
imu_inputs  = 'pos'
imu_outputs = 'pos'
xin, yout = prepare_dataset(u_traj,x_traj,imu_inputs,imu_outputs)

# Training and testing dataset
x_train, y_train, x_test, y_test = split_train_test_traj(xin,yout,nb_traj,training_ratio)

# Get normalization on training dataset (for only u, for all input and all output)
u_mu, u_std, xin_mu, xin_std, yout_mu, yout_std = get_normalization(x_train[:,:,:2],x_train,y_train)

# Get dimensions and window_size
dim_u = u_traj.shape[2]
dim_input, dim_output  = x_train.shape[2], y_train.shape[2]
window_size = int(time_window*fs)
print('- dim_input:',dim_input,', dim_output:',dim_output)
print('- window_size:',window_size)

# Model
model = LSTM(dim_input, dim_hidden, nb_layers, dim_pre_output, dim_output, bidirectional)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fun = torch.nn.MSELoss()          
loss_log = np.array([])

try:
  for epoch in range(nb_epochs):
    for i in range(nb_train):
      input_  = x_train[i,:,:] # theta, phi, theta_dot, phi_dot
      target_ = y_train[i,:,:] # u = uc + ug

      # Normalization
      input_ = (input_ - xin_mu)/xin_std if normalize_input else input_
      target_= (target_- yout_mu)/yout_std if normalize_output else target_

      # Create time-windows for training
      XTrain, YTrain, T_train = model.process_input(input_, window_size, offset=1,
                                                    y=target_, time=t_)
      # Train
      optimizer.zero_grad()
      pred = model(XTrain)
      loss = loss_fun(pred,YTrain)
      loss_log = np.append(loss_log,loss.item())
      loss.backward()
      optimizer.step()
      print("EPOCH: ", epoch, " |  TRAJ:",i, " |  LOSS: ", loss_log[-1])

except KeyboardInterrupt:
    pass

# Losses
torch.cuda.empty_cache()
plt.subplots(figsize=(4,3))
plt.plot(loss_log)
plt.show()

# Check accuracy for one trajectory
XTest, YTest, T_test = approximator.process_input(x_test[i,:,:], window_size, offset=1,
                                          y=y_test[i,:,:], time=t_)
mse_train, mse_test = evaluate_model(approximator,
                                     XTrain=XTrain,YTrain=YTrain,
                                     XTest=XTest,YTest=YTest,
                                     t_train=T_train, t_test=T_test,
                                     vis=1)

# Evaluate on testing set:
init_samples = 300
# Ypred_init,_ = predict(model, X_init,Y_init)
# bias = (Ypred_init.detach().numpy()).mean(0)
fig, axes = plt.subplots(5,5,figsize=(14,10))
ave_mse = plot_predicted_paths(axes,x_test,y_test,model,window_size,init_samples)


# Forward predictions
# pad it with zeros at the beggining
k = 5
N = 1000
train_x = np.pad(x_train[k, :, :], ((N, 0), (0, 0)), mode='constant')
train_y = np.pad(y_train[k, :, :], ((N, 0), (0, 0)), mode='constant')

test_x = np.pad(x_test[k, :, :], ((N, 0), (0, 0)), mode='constant')
test_y = np.pad(y_test[k, :, :], ((N, 0), (0, 0)), mode='constant')

t_pad = np.arange(0,len(train_x)*dt,dt)

ytrue_train, ypred_train, t_train = forward_prediction(model,train_x,train_y,t_pad,window_size)

print(ytrue_train.shape, ypred_train.shape, t_train.shape)

ytrue_test, ypred_test, t_test = forward_prediction(model,test_x,test_y,t_pad,window_size)

mse_train, mse_test =  plot_regression(
                          ytrain=ytrue_train, ytrain_pred=ypred_train, t_train=t_train, 
                          ytest=ytrue_test, ytest_pred=ypred_test,t_test=t_test
                          )


##

# Testing set
# data = pickle.load(open('data/traj_center_100_3s.pkl','rb')) # x: u_theta, u_phi
# t      = data['t']
# x_traj = data['x'] # state vec: theta, phi, theta_dot, phi_dot
# u_traj = data['u'] # u = uc + ug
# nb_traj = len(x_traj)

# # Dataset
# xin_test_3s  = np.concatenate((u_traj[:,:-1,:], x_traj[:,:-1,:2]), axis=2)
# yout_test_3s = x_traj[:,1:,:2]

# Test
# Testing data
# test_x = np.load('data/test_x.npy')
# test_y = np.load('data/test_y.npy')

# test_x, test_y = get_input_output(test_x,test_y,option=1)
# # test_x = (test_x - train_x.mean(0))/train_x.std(0)
# # test_y = (test_y - test_y.mean(0))/test_y.std(0)

# XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
#                                                   y=test_y, time=t)

# mse_test, _ = evaluate_model(trained_model,XTest,YTest,t_train=t_test,vis=1)