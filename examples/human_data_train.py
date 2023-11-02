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

# Load and process real dataset (EMG, IMU)
# data_path = 'data/data_10_56.pkl'
data_path = '/Users/carol/repos/bomi_ws/data/subjects/carol/_01_11_2023/imu/data_19_27.pkl'
data_ = pickle.load(open(data_path,'rb'))
data = Data_processing(data_,degrees=0,downsample_factor=1)
data.load_data()
# data.process_emg(vis=0)

features,target,time =data.get_emgfeatures(data.emgdata,data.desCmd,data.t)
features_filt = data.filter_features()

# t = time
# x = features_filt[:,:4]
# y = target

# With shift
# shift = 5
# t = time[:-shift]
# x = features_filt[:-shift,:4]
# y = target[shift:,:]

# Autoregressive option
t = time[1:]
x = np.hstack((features_filt[:-1,:4], target[:-1,:]))
y = target[1:,:]

x_mu, x_std = x.mean(0), x.std(0)
xn = (x-x_mu)/x_std

ymin, ymax = abs(y.min(0)), abs(y.max(0))
yn = np.where(y>=0,y/ymax,y/ymin)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(data.t_f,data.angles)
# ax.plot(t,xn)
# # ax = fig.add_subplot(212)
# ax.plot(t,yn)
# plt.show()

# x = data.torso_angles
# y = data.features_norm
print('x.shape:',x.shape,', y.shape:',y.shape)

# Train LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_input, dim_output = np.shape(x)[1], np.shape(y)[1]
dim_hidden = 80
nb_layers = 2
dim_pre_output = 40
bidirectional = False

time_window = 1 # sec
window_size = int(time_window*data.fs_features)
offset = 1
print('window_size=',window_size,'| offset=',offset)


#LSTM
approximator = LSTM(dim_input, dim_hidden, nb_layers,
                    dim_pre_output, dim_output, bidirectional).to(device)

train_x, test_x, train_y, test_y, t_train, t_test = split_train_test(xn,yn,train_ratio=0.75,t=t)

XTrain, YTrain, t_train = approximator.process_input(train_x, window_size, offset,
                                                     y=train_y, time=t_train)
# print(XTrain.shape, YTrain.shape, t_train.shape)
XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
                                                     y=test_y, time=t_test)


# Train
nb_epochs = 500
mini_batch_size = 20 # or 30
learning_rate = 1e-3
weight_decay  = 1e-5
training_ratio = 0.75


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
plt.plot(trainer.losses)
plt.show()

mse_train, mse_test = evaluate_model(trained_model,
                                     XTrain=XTrain,YTrain=YTrain,
                                     XTest=XTest,YTest=YTest,
                                     t_train=t_train, t_test=t_test,
                                     vis=1)

# Save model and parameters
params = {"mu": train_x.mean(0), "std": train_x.std(0), "maxs":train_x.max(0),
            "window_size": window_size, "offset": offset,
            "dim_input": dim_input, "dim_output": dim_output, 
            "dim_pre_output": dim_hidden, "nb_layers": nb_layers,
            "bidirectional": bidirectional,
            "mini_batch_size": mini_batch_size, "learning_rate": learning_rate, 
            "weight_decay": weight_decay, "nb_epochs": nb_epochs}     

# save_model(path='data', model=trained_model, params=params)