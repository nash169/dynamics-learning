#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainerLSTM import Trainer
from emg_regression.utils.model_tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data = pickle.load(open('data/traj_center_100_3s.pkl','rb')) # x: u_theta, u_phi
t      = data['t']
u_traj = data['u'] # input vec: u = uc + ug
x_traj = data['x'] # state vec: theta, phi, theta_dot, phi_dot
nb_traj = len(x_traj)
traj_len = len(x_traj[0])

# Set training params
dt = 0.01             # 100 Hz , 100 samples/ second
fs = 1/dt

# Prepare data to make forward predictions
xin, yout, t_ = prepare_dataset(u_traj,
                                x_traj,
                                t,
                                goal='ForwardPred',
                                imu_inputs ='pos',
                                imu_outputs='pos')

# ============================
# Params to set
normalize_input  = True
normalize_states = False
# input_norm = 'min-max'
states_norm = 'min-max'
input_norm = 'std'
# states_norm = 'std'

# LSTM
dim_input, dim_output = xin.shape[-1], yout.shape[-1]
dim_hidden = 80
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 0.05 #s
offset = 1

# train
mini_batch_size = None
learning_rate = 1e-3
weight_decay  = 1e-6
training_ratio = 0.75
nb_epochs = 100

# Get window_size
window_size = int(time_window*fs)

print("============================")
print('normalize_input: ',normalize_input)
print('normalize_states:',normalize_states)
print('\ndim_input:      ',dim_input)
print('dim_output:     ',dim_output) 
print('dim_hidden:     ',dim_hidden)
print('nb_layers:      ',nb_layers)
print('dim_pre_output: ',dim_pre_output) 
print('window_size:    ',window_size)
print('\nmini_batch_size:',mini_batch_size) 
print('learning_rate:  ',learning_rate) 
print('weight_decay:   ',weight_decay) 
print("============================")

# ============================

# Approximator
model = LSTM(dim_input, dim_hidden, nb_layers, dim_pre_output, dim_output, bidirectional).to(device)
model.train()

# Training and testing dataset
#NOTE: not normnalized
x_train, y_train, x_test, y_test = split_train_test_traj(xin,yout,nb_traj,training_ratio,shuffle=True)

x_train_, y_train_, norms = normalize_dataset(x_train,y_train,
                                            normalize_input,normalize_states,
                                            input_norm,states_norm,
                                            norms=None, #it will calculate norms with training set
                                            vis=0)

u_mu , u_std, u_min, u_max, y_mu , y_std, y_min, y_max = norms


XTrain, YTrain = process_input_traj(x_train_,y_train_,model,window_size,traj_len,device)
print(XTrain.shape, YTrain.shape)

## Automatic training =======
trainer = Trainer(model=model, input=XTrain, target=YTrain)
trainer.options(normalize_input=False,
                normalize_output=False,
                epochs=nb_epochs,
                batch=mini_batch_size, #set to None if don't want batches
                shuffle=False,
                record_loss=True,
                stateful_train=True
                )

trainer.optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
trainer.train()

trained_model = trainer.model
loss_log = trainer.epoch_losses
model = trainer.model

# Losses
torch.cuda.empty_cache()
plt.subplots(figsize=(4,3))
plt.plot(loss_log)
plt.show()

# model evaluation
model.eval()

# Path prediction on Training and testing sets
with torch.no_grad():
    i = 2

    # Normalize the testing set
    x_test_, y_test_, _ = normalize_dataset(x_test,y_test,
                                            normalize_input,normalize_states,
                                            input_norm,states_norm,
                                            norms,
                                            vis=0)

    # Check Train and Test accuracy for one trajectory

    # Initial bias 
    init_samples = 300 #300
    X_init = torch.zeros(init_samples,window_size,dim_input).to(device)
    Y_init = torch.zeros(init_samples,dim_output).to(device)
    Ypred_init, _ = predict(model, X_init,Y_init)
    bias = (Y_init - Ypred_init).mean(0).cpu().detach().numpy()

    #build windows
    XTrain, YTrain, T_train = model.process_input(x_train_[i,:,:], window_size, offset=1,
                                                    y=y_train_[i,:,:], time=t_)
    

    XTest, YTest, T_test = model.process_input(x_test_[i,:,:], window_size, offset=1,
                                            y=y_test_[i,:,:], time=t_)
  

    # Evauate tracking accuracy
    mse_train, mse_test = evaluate_model(model,
                                        XTrain=XTrain,YTrain=YTrain,t_train=T_train,
                                        XTest=XTest,YTest=YTest,t_test=T_test,
                                        bias=bias,
                                        vis=1)

    # Evaluate paths on training and testing set:
    fig, axes = plt.subplots(5,5,figsize=(14,10))
    ave_mse = plot_predicted_paths(axes,x_train_,y_train_,model,window_size,init_samples)

    fig, axes = plt.subplots(5,5,figsize=(14,10))
    ave_mse = plot_predicted_paths(axes,x_test_,y_test_,model,window_size,init_samples)

    x = x_train_[i,:,:]
    y = y_train_[i,:,:]
    batch_size = 1
    ypred = forward_prediction(model,x,y,
                        window_size,
                        batch_size,
                        correct_bias=0)
