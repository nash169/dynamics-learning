#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
x = np.load('data/train_x.npy') # x: u_theta, u_phi
y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Option 1: input: u1,u2, output: theta, phi
# Option 2: input: theta(k), phi(k), u1(k), u2(k), output: theta(k+1), phi(k+1)


dt = 0.01             # 100 Hz , 100 samples/ second
fs = 1/dt
t = np.arange(0,len(x)*dt,dt)

normalize_input  = True
normalize_output = True

# LSTM
dim_hidden = 20
nb_layers = 2
dim_pre_output = 20
bidirectional = False
time_window = 0.5 #s
offset = 1

# train
mini_batch_size = 20
learning_rate = 1e-2
weight_decay  = 1e-6
nb_epochs = 200
training_ratio = 0.75

# TODO create function to do this automatically 

if normalize_input:
    x_mu, x_std = x.mean(0), x.std(0)
    x = (x-x_mu)/x_std

if normalize_output:
    y_mu, y_std = y.mean(0), y.std(0)
    y = (y-y_mu)/y_std
    # ymin, ymax = abs(target_.min(0)), abs(target_.max(0))
    # target_ = np.where(target_>=0, target_/ymax, target_/ymin)

input_  = np.append(x[:-1,:], y[:-1,:2],axis=1) # u1(k), u2(k), theta(k), phi(k)
target_ = y[1:,:2] 
t_ = t[1:]

# plt.plot(input_)
# plt.plot(target_)
# plt.show()

dim_input, dim_output  = input_.shape[1], target_.shape[1]
window_size = int(time_window*fs)
print('- dim_input:',dim_input,', dim_output:',dim_output)
print('- window_size:',window_size)


#LSTM
approximator = LSTM(dim_input, dim_hidden, nb_layers,
                    dim_pre_output, dim_output, bidirectional)

train_x, test_x, train_y, test_y, t_train, t_test = split_train_test(input_,target_,
                                                                    training_ratio,
                                                                    t=t_)
                                                                    
XTrain, YTrain, T_train = approximator.process_input(train_x, window_size, offset=1,
                                                    y=train_y, time=t_train)

XTest, YTest, T_test = approximator.process_input(test_x, window_size, offset=1,
                                                    y=test_y, time=t_test)

print(XTrain.shape, YTrain.shape, T_train.shape)


# Train
optimizer = torch.optim.Adam(approximator.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fun = torch.nn.MSELoss()          
loss_log = np.array([])

try:
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        pred = approximator(XTrain)
        loss = loss_fun(pred,YTrain)
        loss_log = np.append(loss_log,loss.item())
        loss.backward()
        optimizer.step()
        print("EPOCH: ", epoch, " |  LOSS: ", loss_log[-1])
except KeyboardInterrupt:
    pass

# Losses
torch.cuda.empty_cache()
plt.subplots(figsize=(4,3))
plt.plot(loss_log)
plt.show()

trained_model = approximator 

mse_train, mse_test = evaluate_model(trained_model,
                                     XTrain=XTrain,YTrain=YTrain,
                                     XTest=XTest,YTest=YTest,
                                     t_train=T_train, t_test=T_test,
                                     vis=1)

# Forward predictions
model = trained_model
x_dim = np.shape(x)[1]

# Unseen predictions on Training set
ytrain = train_y[window_size:,:]
x_tw = train_x[:window_size,:x_dim]
y_tw = train_x[:window_size,x_dim:]
ypred_k = train_x[window_size,x_dim:]
ypred = y_tw

for i in range(window_size,len(train_y)):
  # Get new emg window
  x_i = train_x[i,:x_dim]
  x_tw = np.vstack((x_tw,x_i))[-window_size:,:]
  X_tw = torch.from_numpy(x_tw).unsqueeze(0).float().to(device)

  # Get last predictions
  y_i = ypred_k
  y_tw = np.vstack((y_tw,y_i))[-window_size:,:]
  Y_tw = torch.from_numpy(y_tw).unsqueeze(0).float().to(device)

  # Concatenate to create network input (1,window,6)
  input_tw = torch.cat((X_tw,Y_tw),dim=2)

  # Normalize input
  # input_tw = input_tw.sub_(X_mu).div_(X_std)

  # Make prediction
  ypred_k = model(input_tw).cpu().detach().numpy()
  ypred = np.vstack((ypred,ypred_k))

# ypred = ypred
ypred_train = ypred[window_size:,:]
t_train_ = t_train[window_size:]

print(ytrain.shape, ypred_train.shape, t_train_.shape)

# Unseen predictions on Testing set
ytest = test_y
x_tw = train_x[-window_size:,:x_dim]
y_tw = train_x[-window_size:,x_dim:]
ypred_k = test_x[0,x_dim:]
ypred = y_tw

for i in range(len(test_y)):
  # Get new emg window
  x_i = test_x[i,:x_dim]
  x_tw = np.vstack((x_tw,x_i))[-window_size:,:]
  X_tw = torch.from_numpy(x_tw).unsqueeze(0).float().to(device)

  # Get last predictions
  y_i = ypred_k
  y_tw = np.vstack((y_tw,y_i))[-window_size:,:]
  Y_tw = torch.from_numpy(y_tw).unsqueeze(0).float().to(device)

  # Concatenate to create network input (1,window,6)
  input_tw = torch.cat((X_tw,Y_tw),dim=2)

  # Normalize input
  # input_tw = input_tw.sub_(X_mu).div_(X_std)

  # Make prediction
  ypred_k = model(input_tw).cpu().detach().numpy()
  ypred = np.vstack((ypred,ypred_k))

ypred_test = ypred[window_size:,:]
t_test_ = t_test

print(ytest.shape, ypred_test.shape, t_test_.shape)


mse_train, mse_test =  plot_regression(ytrain=ytrain, ytrain_pred=ypred_train, t_train=t_train_, ytest=ytest, ytest_pred=ypred_test,t_test=t_test_)



##

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