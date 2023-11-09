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
len_traj = len(x_traj[0])

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

normalize_input  = 1
normalize_states = 0

# LSTM
dim_hidden = 50
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 0.05 #s
offset = 1

# train
mini_batch_size = 15
learning_rate = 1e-3
weight_decay  = 1e-6
training_ratio = 0.75
nb_epochs = 50

# Testing
# batch_size = len_traj

# Prepare data to make forward predictions
imu_inputs  = 'pos'
imu_outputs = 'pos'
xin, yout = prepare_dataset(u_traj,x_traj,imu_inputs,imu_outputs)
t_ = t[1:]

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
hidden = model.initialize_states(batch_size=len_traj-window_size)

# Prepare dataset
XTrain, YTrain = process_input_traj(x_train,y_train,model,window_size,len_traj,device)

mini_batch_size = len_traj-window_size
# mini_batch_size = 10

try:
  for epoch in range(nb_epochs):
    for i in range(len(XTrain)):
      
      (h0, c0) = model.initialize_states(batch_size=mini_batch_size)
      XTrain_traj, YTrain_traj = XTrain[i], YTrain[i]
      
      # Create loader for batch training
      torch_dataset = torch.utils.data.TensorDataset(XTrain_traj, YTrain_traj)
      loader = torch.utils.data.DataLoader(dataset=torch_dataset,batch_size=mini_batch_size,shuffle=False,num_workers=0)
  
      # Train
      batch_loss = []
      for iter, (X_batch, Y_batch) in enumerate(loader):  # for each training step                              
        optimizer.zero_grad()
        pred, (h0, c0) = model.forward_predict(X_batch,(h0, c0))
        loss = loss_fun(pred,Y_batch)
        loss.backward()
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

# check initial bias, offset 
init_samples = 300
X_init = torch.zeros(init_samples,window_size,dim_input).to(device)
Y_init = torch.zeros(init_samples,dim_output).to(device)
Ypred_init, _ = predict(model, X_init,Y_init)
bias = (Y_init - Ypred_init).mean(0).cpu().detach().numpy()

# Check accuracy for one trajectory
i = 1
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

k = 2
u   = x_train[k,:,:2]
y = y_train[k,:,:2]
batch_size = 5

ypred = forward_prediction(model,u,y,
                   window_size,
                   batch_size)

# Forward predictions
# TODO: Try foward predictions on Training set, I should obtain something similar to the case in which 
# my input is uin + yin(k-1)


def inference(x_train,y_train,k,batch_size,init_samples):
  sample_size = window_size + (batch_size-1)

  train_x = np.pad(x_train[k,:,:], ((init_samples,0), (0,0)), mode='constant').astype(float)
  train_y = np.pad(y_train[k,:,:], ((init_samples,0), (0,0)), mode='constant').astype(float)
  t_pad = np.arange(0,len(train_x)*dt,dt)

  u_tw     = train_x[:sample_size,:2]
  yin_tw   = train_x[:sample_size,2:]
  ypred_k  = train_y[sample_size-1]
  input_tw = np.hstack((u_tw,yin_tw))

  X_batch,Y_batch,_ = model.process_input(x=input_tw,window_size=window_size,offset=1)
  print(f'Í will wait for {sample_size} samples')

  ypred = np.array([]).reshape(0,2)
  model.eval()
  # hidden = model.initialize_states(batch_size)

  for i in range(sample_size, len(train_x)):

    u_i = train_x[i,:2]
    yin_i = ypred_k
    input_i = np.append(u_i,yin_i)[np.newaxis]

    input_tw = np.vstack((input_tw, input_i))[-window_size:,:]
    Input_tw = torch.from_numpy(input_tw).unsqueeze(0).float().to(device)

    X_batch = torch.cat((X_batch,Input_tw))[-batch_size:]
    # Ypred_tw, hidden  = model.forward_predict(X_batch, hidden)
    # h_0, c_0 = hidden
    # h_0.detach_(), c_0.detach_()    # remove gradient information
    # hidden = (h_0, c_0)
    Ypred_tw  = model(X_batch)
    ypred_tw  = Ypred_tw.cpu().detach().numpy()
    ypred_k   = ypred_tw[-1]
    ypred = np.vstack((ypred, ypred_k))

  ytrain   = train_y[sample_size:]
  t_train_ = t_pad[sample_size:]
  # mse_train, mse_test =  plot_regression(
  #                           ytrain=ytrain, ytrain_pred=ypred, t_train=t_train_ )

  return ypred, t_train_

k = 2
batch_size = 10
init_samples = 0

inference(x_train,y_train,k,batch_size,init_samples)


batch_sizes = np.arange(1,280,20)
y1, t1 = [],[]
for batch_size in batch_sizes:
  print(batch_size)
  ypred_, tp_ = inference(x_train,y_train,k,batch_size,init_samples)
  y1.append(ypred_)
  t1.append(tp_)

colormap = plt.cm.get_cmap('viridis', len(batch_sizes))

fig, ax = plt.subplots(2,1)
for i, batch_size in enumerate(batch_sizes):
    color = colormap(i / len(batch_sizes))
    for m in range(2):
      if i==0:
        ax[m].plot(t_, y_train[k,:,:][:,m], label=f'Batch Size {batch_size}',color=color)
      ax[m].plot(t1[i], y1[i][:,m], label=f'Batch Size {batch_size}',color=color)
      ax[m].grid()
ax[0].legend()
plt.show()

##### =============================
# The correct and expected predictions
XTrain, YTrain, T_train = model.process_input(x_train[k,:,:], window_size, offset=1,
                                          y=y_train[k,:,:], time=t_)

ypred_all = model(XTrain).cpu().detach().numpy()
plt.plot(T_train,YTrain.cpu().detach().numpy())
plt.plot(T_train,ypred_all)
plt.show()

##### =============================
# What's the effect of batch length in prediction?  
 
batch_size = 5
total_steps = len(XTrain) - batch_size + 1


ypred = np.array([]).reshape(0,2)
t2 = T_train[batch_size-1:]
for i in range(total_steps):
  
  X_batch = XTrain[i:i+batch_size]


  ypred_ = model(X_batch)[-1].cpu().detach().numpy()[np.newaxis]
  ypred = np.vstack((ypred,ypred_))

plt.plot(t_,y_train[k,:,:])

plt.plot(t2,ypred)
plt.show()





# ytrue_train, ypred_train, t_train = forward_prediction(model,train_x,train_y,t_pad,window_size,bias)


# ytrue_test, ypred_test, t_test = forward_prediction(model,test_x,test_y,t_pad,window_size,bias)

# mse_train, mse_test =  plot_regression(
#                           ytrain=ytrue_train, ytrain_pred=ypred_train, t_train=t_train, 
#                           ytest=ytrue_test, ytest_pred=ypred_test,t_test=t_test
#                           )


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