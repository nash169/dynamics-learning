#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data = pickle.load(open('data/traj_center_100_3s.pkl','rb')) # x: u_theta, u_phi
t      = data['t']
x_traj = data['x'] # state vec: theta, phi, theta_dot, phi_dot
u_traj = data['u'] # u = uc + ug
uc_traj = data['uc']
ug_traj = data['ug']
nb_traj = len(x_traj)
traj_len = len(x_traj[0])

# Dataset
# xin  = np.concatenate((u_traj[:,:-1,:], x_traj[:,:-1,:2]), axis=2)
# yout = x_traj[:,1:,:2]

# System identification
# xin  = u_traj[:,:,:]
# yout = x_traj[:,:,:2]
# t_ = t

# Set training params
dt = 0.01             # 100 Hz , 100 samples/ second
fs = 1/dt



# LSTM
normalize_input  = 0
normalize_states = 0

dim_hidden = 50
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 0.05 #s
offset = 1

# train
mini_batch_size = 15
learning_rate = 1e-3
weight_decay  = 1e-5
training_ratio = 0.75
nb_epochs = 50

# Testing
# batch_size = traj_len

# Prepare data to make forward predictions
xin, yout, t_ = prepare_dataset(u_traj,x_traj,t,
                            goal='ForwardPred',
                            imu_inputs ='pos',
                            imu_outputs='pos')

# Training and testing dataset
x_train, y_train, x_test, y_test = split_train_test_traj(xin,yout,nb_traj,training_ratio,shuffle=True)

# Get normalization on training dataset (for only u, for all input and all output)
u_mu, u_std, xin_mu, xin_std, yout_mu, yout_std = get_normalization(x_train[:,:,:2],x_train,y_train)

# Get dimensions and window_size
dim_u = u_traj.shape[2]
dim_input, dim_output  = x_train.shape[2], y_train.shape[2]
window_size = int(time_window*fs)
print('- dim_input:',dim_input,', dim_output:',dim_output)
print('- window_size:',window_size)

# Normalize
if normalize_input:
  # For this type of data, min/max normalization is best result
  umax, umin = abs(x_train[:,:,:2].max(0).max(0)), abs(x_train[:,:,:2].min(0).min(0))
  for i in range(dim_u):
    x_train[:,:,i] = np.where(x_train[:,:,i]>=0, x_train[:,:,i]/umax[i],x_train[:,:,i]/umin[i])
    x_test[:,:,i] = np.where(x_test[:,:,i]>=0, x_test[:,:,i]/umax[i], x_test[:,:,i]/umin[i])
  print("Input normalized")

if normalize_states:
  ymax, ymin = abs(y_train.max(0).max(0)), abs(y_train.min(0).min(0))
  for i in range(dim_output):
    y_train[:,:,i] = np.where(y_train[:,:,i]>=0, y_train[:,:,i]/ymax[i], y_train[:,:,i]/ymin[i])
    y_test[:,:,i]  = np.where(y_test[:,:,i] >=0, y_test[:,:,i]/ ymax[i], y_test[:,:,i]/ ymin[i])
  print("States normalized")


# Model
model = LSTM(dim_input, dim_hidden, nb_layers, dim_pre_output, dim_output, bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fun = torch.nn.MSELoss()          
loss_log = np.array([])
# hidden = model.initialize_states(batch_size=traj_len-window_size)

# Prepare dataset
XTrain, YTrain = process_input_traj(x_train,y_train,model,window_size,traj_len,device)


mini_batch_size = 10 #must be divisible by len(x) if dont want to drop the last
mini_batch_size = traj_len-window_size
remainder = mini_batch_size - (len(XTrain[-3])% mini_batch_size)
pad_xbatch = torch.ones(remainder,window_size,dim_input, device=device)
pad_ybatch = torch.ones(remainder,dim_output, device=device)
print(f"Samples discarded per trajectory: {len(XTrain[1]) % mini_batch_size}")

## Automatic training =======
model = LSTM(dim_input, dim_hidden, nb_layers, dim_pre_output, dim_output, bidirectional).to(device)
trainer = Trainer(model=model, input=XTrain, target=YTrain)
trainer.options(normalize_input=True,
                input_norm='standard',
                normalize_output=False,
                target_norm='min-max',
                epochs=nb_epochs,
                batch=mini_batch_size,
                shuffle=False,
                record_loss=True)

trainer.optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
trainer.train()
trained_model = trainer.model
loss_log = trainer.epoch_losses
## ===========

XTrain.requires_grad = True
try:
  for epoch in range(nb_epochs):
    for i in range(len(XTrain)):
      
      (h0, c0) = model.initialize_states(batch_size=mini_batch_size)
      XTrain_traj, YTrain_traj = XTrain[i], YTrain[i]
      
      # Create loader for batch training
      torch_dataset = torch.utils.data.TensorDataset(XTrain_traj, YTrain_traj)
      loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                          batch_size=mini_batch_size,
                                          shuffle=False,
                                          drop_last=False, #otherwise gives error for h0,c0 shape
                                          num_workers=0)

      # Train
      batch_loss = []
      for iter, (X_batch, Y_batch) in enumerate(loader):  # for each training step   
        if len(X_batch) != mini_batch_size:
          X_batch = torch.cat((X_batch, pad_xbatch * X_batch[-1,-1,:]))
          Y_batch = torch.cat((Y_batch, pad_ybatch * Y_batch[-1,:]))

        optimizer.zero_grad()
        pred, (h0, c0) = model(X_batch,(h0, c0))
        loss = loss_fun(pred,Y_batch)
        loss.backward() # computes  gradients of the loss with respect to all tensors that have requires_grad=True.
        optimizer.step()

        h0.detach_(), c0.detach_()    # remove gradient information

        # Record losses
        loss_batch = loss.item()
        batch_loss.append(loss_batch)
        loss_log = np.append(loss_log,loss_batch)

      ave_traj_loss = np.array(batch_loss).mean()
      print("EPOCH: ", epoch, " |  TRAJ:",i, " |  LOSS: ", ave_traj_loss)
      # print("EPOCH: ", epoch, " |  TRAJ:",i, " |  LOSS: ", loss.item())

except KeyboardInterrupt:
    pass



# Losses
torch.cuda.empty_cache()
plt.subplots(figsize=(4,3))
plt.plot(loss_log)
plt.show()

model.eval()

with torch.no_grad():
  # check initial bias, offset 
  init_samples = 300
  X_init = torch.zeros(init_samples,window_size,dim_input).to(device)
  Y_init = torch.zeros(init_samples,dim_output).to(device)
  Ypred_init, _ = predict(model, X_init,Y_init)
  bias = (Y_init - Ypred_init).mean(0).cpu().detach().numpy()

  # Check accuracy for one trajectory
  i = 2
  XTrain, YTrain, T_train = model.process_input(x_train[i,:,:], window_size, offset=1,
                                                y=y_train[i,:,:], time=t_)
  XTest, YTest, T_test = model.process_input(x_test[i,:,:], window_size, offset=1,
                                            y=y_test[i,:,:], time=t_)
  mse_train, mse_test = evaluate_model(model,
                                      XTrain=XTrain,YTrain=YTrain,t_train=T_train,
                                      XTest=XTest,YTest=YTest,t_test=T_test,
                                      bias=bias,
                                      vis=1)

  # Evaluate paths on training and testing set:
  fig, axes = plt.subplots(5,5,figsize=(14,10))
  ave_mse = plot_predicted_paths(axes,x_train,y_train,model,window_size,init_samples=300)


  # bias = (Ypred_init.detach().numpy()).mean(0)
  fig, axes = plt.subplots(5,5,figsize=(14,10))
  ave_mse = plot_predicted_paths(axes,x_test,y_test,model,window_size,init_samples)

with torch.no_grad():
  k = 1
  x = x_test[k,:,:]
  y = y_test[k,:,:2]
  batch_size = 1

  ypred = forward_prediction(model,
                             x,y,
                            window_size,
                            batch_size,
                            correct_bias=True)





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