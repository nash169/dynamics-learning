#!/usr/bin/env python

import torch
import numpy as np
import pickle
from emg_regression.utils.data_processing import Data_processing
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import split_train_test, evaluate_model, save_model
import matplotlib.pyplot as plt

# Load and process real dataset (EMG, IMU)
data_path = 'data/data_10_56.pkl'
data = pickle.load(open(data_path,'rb'))
data = Data_processing(data,degrees=0,downsample_factor=20)
data.load_data()
data.pre_process(vis=0, filt=1)

x = data.features_norm
y = data.torso_angles
# x = data.torso_angles
# y = data.features_norm
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
weight_decay  = 1e-6
training_ratio = 0.75


#LSTM
approximator = LSTM(dim_input, dim_hidden, nb_layers,
                    dim_pre_output, dim_output, bidirectional)

train_x, test_x, train_y, test_y = split_train_test(x,y,train_ration=0.75,t=None)

XTrain, YTrain, t_train = approximator.process_input(train_x, window_size, offset,
                                                     y=train_y, time=None)
# print(XTrain.shape, YTrain.shape, t_train.shape)
XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
                                                     y=test_y, time=None)


# Train
trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize=True,
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