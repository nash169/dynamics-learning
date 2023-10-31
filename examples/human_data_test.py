#!/usr/bin/env python

import torch
import numpy as np
import pickle
from emg_regression.utils.data_processing import Data_processing
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import split_train_test, evaluate_model, save_model, plot_regression
import matplotlib.pyplot as plt

# Load and process real dataset (EMG, IMU)
data_path = 'data/data_10_56.pkl'
data = pickle.load(open(data_path,'rb'))
data = Data_processing(data,degrees=0,downsample_factor=20)
data.load_data()
data.pre_process(vis=0, filt=1)

x_ = data.features_norm
y_ = data.torso_angles
t_ = data.t

x = np.hstack((x_,y_))[:-1,:]
y = y_ [1:,:]
t = t_ [1:]
# x = data.torso_angles
# y = data.features_norm
print('x.shape:',x.shape,', y.shape:',y.shape)

# Normalize
# y = (y - y_mu)/y_std

# Train LSTM
dim_input, dim_output = np.shape(x)[1], np.shape(y)[1]
dim_hidden = 20
nb_layers = 2
dim_pre_output = 20
bidirectional = False

time_window = 1 # sec
window_size = int(time_window*data.fs)
offset = 1
print('window_size=',window_size,'| offset=',offset)


#LSTM
approximator = LSTM(dim_input, dim_hidden, nb_layers,
                    dim_pre_output, dim_output, bidirectional)

train_x, test_x, train_y, test_y, t_train, t_test = split_train_test(x,y,train_ratio=0.75,t=t_)

# Normalize here
x_mu, x_std = train_x.mean(0), train_x.std(0)
train_x = (train_x - x_mu)/x_std

# Form time-windows
XTrain, YTrain, t_train = approximator.process_input(train_x, window_size, offset,
                                                     y=train_y, time=t_train)
# print(XTrain.shape, YTrain.shape, t_train.shape)

# Train
nb_epochs = 200
mini_batch_size = 100 # or 30
learning_rate = 1e-2
weight_decay  = 1e-6
training_ratio = 0.75

trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize=False,
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

# Normalize here
test_x_ = (test_x - x_mu)/x_std
XTest, YTest, T_test = approximator.process_input(test_x_, window_size, offset,
                                                     y=test_y, time=t_test)

mse_train, mse_test = evaluate_model(trained_model,
                                     XTrain=XTrain,YTrain=YTrain,
                                     XTest=XTest,YTest=YTest,
                                     vis=1)

# TRY to evaluate on testing set by giving back prediction a input.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = trained_model
test_input = test_x[:,:4]

X_mu, X_std = torch.from_numpy(x_mu).float(), torch.from_numpy(x_std).float()
# Ypred_test = torch.empty(window_size,y.shape[1]).float().to(device) #must start with zeros
Ypred_test = YTrain[-window_size:] # know last 100 positions
test_input = np.vstack((train_x[-window_size:,:4], test_input))

for k in range(window_size-1,len(test_input)):
    x_tw = test_input[k-window_size+1:k+1,:]
    X_tw = torch.from_numpy(x_tw).unsqueeze(0).float()

    # Get last predictions
    Y_tw = Ypred_test[-window_size:,:].float()

    # Concatenate to create network input (1,window,6)
    input_tw = torch.cat((X_tw,Y_tw.unsqueeze(0)),dim=2)

    # Normalize input
    input_tw = input_tw.sub_(X_mu).div_(X_std)

    # Make prediction
    ypred_k = model(input_tw)
    Ypred_test = torch.cat((Ypred_test,ypred_k))

ypred_test = Ypred_test.detach().numpy()

ypred = ypred_test[window_size+1:,:]
ytrue = test_y
tpred = t_test[:-1]
# tpred = tpred - tpred[0]
mse   = plot_regression(ytrue,ypred,tpred)


# Save model and parameters
params = {"mu": train_x.mean(0), "std": train_x.std(0), "maxs":train_x.max(0),
            "window_size": window_size, "offset": offset,
            "dim_input": dim_input, "dim_output": dim_output, 
            "dim_pre_output": dim_hidden, "nb_layers": nb_layers,
            "bidirectional": bidirectional,
            "mini_batch_size": mini_batch_size, "learning_rate": learning_rate, 
            "weight_decay": weight_decay, "nb_epochs": nb_epochs}     

# save_model(path='data', model=trained_model, params=params)