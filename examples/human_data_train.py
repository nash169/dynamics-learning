#!/usr/bin/env python

import numpy as np
import pickle
from emg_regression.utils.data_processing import Data_processing
from emg_regression.approximators.lstm import LSTM

# Load and process real dataset (EMG, IMU)
data_path = 'data/data_10_56.pkl'
data = pickle.load(open(data_path,'rb'))
data = Data_processing(data,degrees=0,downsample_factor=20)
data.load_data()
data.pre_process(vis=0, filt=1)

x = data.features_norm
y = data.torso_angles
print('x.shape:',x.shape,', y.shape:',y.shape)

# Train LSTM
dim_input, dim_output = np.shape(x)[1], np.shape(y)[1]
dim_hidden = 20
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 1 # sec
window_size = int(time_window*data.fs)
offset = int(window_size*0.1)
print('window_size=',window_size,'| offset=',offset)

nb_epochs = 300
mini_batch_size = 50 # or 30
learning_rate = 1e-3
weight_decay  = 1e-3
training_ratio = 0.75

model = LSTM(dim_input, dim_hidden, nb_layers,
            dim_pre_output,dim_output,bidirectional)

model, losses = model.train_model(model,x,y,training_ratio,window_size,offset,
                nb_epochs,mini_batch_size,learning_rate,weight_decay,data.t)

model.plot_loss(losses)
model.plot_prediction()

# Save model and parameters
model.save_model(path='data', model=model)