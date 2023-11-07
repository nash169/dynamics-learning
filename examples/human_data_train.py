#!/usr/bin/env python
import torch
import numpy as np
import pickle
from emg_regression.utils.data_processing import Data_processing
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import split_train_test, evaluate_model, save_model
import matplotlib.pyplot as plt
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'- Device: {device}')

# Load and process real dataset (EMG, IMU)
# data_path = '/Users/carol/repos/bomi_ws/data/subjects/carol/_01_11_2023/imu/data_02_27.pkl'
data_path = './data/trunk_data/data_10_56.pkl'
data_ = pickle.load(open(data_path,'rb'))
data_['trialID']
data = Data_processing(data_,degrees=0,downsample_factor=1)
data.load_data()
fs = data.fs

# extract features and filter them
features,target,time =data.get_emgfeatures(data.emgdata,data.desCmd,data.t)
features_filt = data.filter_features()
fs_features = data.fs_features

# Set type of model
model_type = 'LSTM'
# model_type = 'Feedback LSTM'

# Normal LSTM (x->y)
t = time
x = features_filt[:,:]
y = target

# With shift
# shift = 5
# t = time[:-shift]
# x = features_filt[:-shift,:4]
# y = target[shift:,:]

# Autoregressive option
# t = time[1:]
# x = np.hstack((features_filt[:-1,:4], target[:-1,:]))
# y = target[1:,:]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(data.t_f,data.angles)
# ax.plot(t,x)
# # ax = fig.add_subplot(212)
# ax.plot(t,y)
# plt.show()

# ================================================
# Parameters to set
input_features   = 'nzc+rms' # ['rms','nzc+rms','raw']
filt_features    = True
normalize_input  = True
normalize_output = True

dim_hidden = 20
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 1      # sec
mini_batch_size = 50 # or 30
learning_rate = 1e-3
weight_decay  = 1e-5

nb_epochs      = 200
training_ratio = 0.75

# ================================================
# Define approximator, input and output
if filt_features:
    features = features_filt    
if input_features == 'rms':
    features = features[:,:4]

if input_features == 'raw':
    features = data.emgdata
    target = data.desCmd
    time = data.t
    fs_ = fs
else:
    fs_ = fs_features

if normalize_input:
    x_mu, x_std = features.mean(0), features.std(0)
    features = (features-x_mu)/x_std
if normalize_output:
    ymin, ymax = abs(target.min(0)), abs(target.max(0))
    target = np.where(target>=0, target/ymax, target/ymin)

if model_type == 'Feedback LSTM':
    input_  = np.hstack((features[:-1,:], target[:-1,:]))
    output_ = target[1:,:]
else:
    input_  = features
    output_ = target

dim_input, dim_output  = input_.shape[1], output_.shape[1]
window_size = int(time_window*fs_)
print('- dim_input:',dim_input,', dim_output:',dim_output)
print('- window_size:',window_size)

approximator = LSTM(dim_input, dim_hidden, nb_layers,
                dim_pre_output, dim_output, bidirectional).to(device)

train_x, test_x, train_y, test_y, t_train, t_test = split_train_test(input_,output_,
                                                                    training_ratio,
                                                                    t=time)

XTrain, YTrain, T_train = approximator.process_input(train_x, window_size, offset=1,
                                                    y=train_y, time=t_train)

XTest, YTest, T_test = approximator.process_input(test_x, window_size, offset=1,
                                                    y=test_y, time=t_test)
# print(XTrain.shape, YTrain.shape, t_train.shape)

# ================================================
# Train
trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize_input=False,
                normalize_output=False,
                epochs=nb_epochs,
                batch=mini_batch_size,
                shuffle=False,
                record_loss=True)

trainer.optimizer = torch.optim.Adam(approximator.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
trainer.train()
trained_model = trainer.model

# Losses
torch.cuda.empty_cache()
plt.subplots(figsize=(4,3))
plt.plot(trainer.losses)
plt.show()

mse_train, mse_test = evaluate_model(trained_model,
                                     XTrain=XTrain,YTrain=YTrain,
                                     XTest=XTest,YTest=YTest,
                                     t_train=T_train, t_test=T_test,
                                     vis=1)

# Save model and parameters
params ={"mu": x_mu, "std": x_std, 
        "ymin": ymin, "ymax": ymax,
        "window_size": window_size,
        "dim_input": dim_input, "dim_output": dim_output,
        "dim_pre_output": dim_hidden, "nb_layers": nb_layers,
        "bidirectional": bidirectional,
        "mini_batch_size": mini_batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "nb_epochs": nb_epochs}


# save_model(path='data', model=trained_model, params=params)