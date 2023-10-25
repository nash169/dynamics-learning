#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.approximators.lstm_ds import LSTM_DS
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import get_input_output, evaluate_model, compute_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data_x = np.load('data/train_x.npy') # x: u_theta, u_phi
data_y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Option 1: input: u1,u2, output: theta, phi
# Option 2: input: theta(k), phi(k), u1(k), u2(k), output: theta(k+1), phi(k+1)
train_x, train_y = get_input_output(data_x,data_y,option=2)
train_y = (train_y - train_y.mean(0))/train_y.std(0)
dt = 0.01             # 100 Hz , 100 samples/ second
t = np.arange(0,len(train_x),dt)

fig = plt.figure()
fig.add_subplot(211).plot(train_x)
fig.add_subplot(212).plot(train_y)
plt.show()


# DS-LSTM
dim_input  = train_x.shape[1]
dim_output = train_y.shape[1]
num_layers = 2
dim_hidden = 20
dim_pre_output = 20
bidirectional = False
window_size = 15
offset = 8

# train
num_epochs = 200
learning_rate = 1e-3
weight_decay  = 1e-5
training_ratio = 1
mini_batch_size = 10

# Model
approximator = LSTM(dim_input, dim_hidden, num_layers,
                    dim_pre_output, dim_output, bidirectional)

XTrain, YTrain, t_train = approximator.process_input(train_x, window_size, offset,
                                                    y=train_y, time=t)
# print(XTrain.shape, YTrain.shape, t_train.shape)


# Train
trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize=True,
                epochs=num_epochs,
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
mse_train, _ = evaluate_model(trained_model,XTrain,YTrain,vis=1)


""" This method might work!! Compare with input emg"""

# Evaluation 
model = trained_model

model.eval()

# input vector
# Option 2: input: [theta(k), phi(k), u1(k), u2(k)], output: theta(k+1), phi(k+1)
# Get the emg signal
# Utest = torch.from_numpy(train_x[:,2:])
Xtest = XTrain[:,:,:2]
Utest = XTrain[:,:,2:]

Ypred = torch.zeros((window_size, dim_output))
# Ypred = YTrain[:window_size,:]
Xtest = torch.zeros((1, dim_input))

# must add the offset and fix Xtest
for k in range(0,len(Utest)-window_size-1):
    # input = [theta_k, phi_k, u1_k, u2_k] 
    Utest_window = Utest[k+window_size+1]
    Ypred_window = YTrain[k:k+window_size,:] #Ypred[-window_size:] 

    XTest_k = torch.hstack((Ypred_window,Utest_window)).unsqueeze(0).float()
    with torch.no_grad():
        ypred_k = model(XTest_k.to(device))
    Ypred = torch.vstack((Ypred,ypred_k))
    Xtest = torch.vstack((Xtest,XTest_k[0,-1,:]))

Ypred = Ypred[window_size-1:,:]
ytrue = YTrain[:,:2].detach().numpy()
ypred =  Ypred.detach().numpy()

m = window_size-offset +2
# m = 0
tt1,tt2 = t_train,t_train[m:]
fig, ax = plt.subplots(2,1)
for j in range(2):
    ax[j].plot(ytrue[m:,j],'r',label='True')
    ax[j].plot(ypred[:,j],'b',label='Predicted')
ax[0].legend()
plt.show()

compute_loss(Ypred, YTrain[window_size:,:2])


#########

# Trainer
loss_fun  = torch.nn.MSELoss()
optimizer = torch.optim.Adam(approximator.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_log = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred = approximator(train_x)
    loss = loss_fun(pred,train_y)
    loss_log.append(loss.item())
    loss.backward()
    optimizer.step()

loss_log = np.array(loss_log)
# Train
trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
trainer.options(normalize=True,
                epochs=num_epochs,
                batch=mini_batch_size,
                shuffle=False,
                record_loss=True)

trainer.optimizer = torch.optim.Adam(approximator.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)

loss = trainer.train()
trained_model = trainer.model
plt.plot(trainer.losses)
plt.show()

mse_train, _ = evaluate_model(trained_model,XTrain,YTrain,vis=1)

# Test
# Testing data
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')

test_x, test_y = get_input_output(test_x,test_y,option=1)
# test_x = (test_x - train_x.mean(0))/train_x.std(0)
# test_y = (test_y - test_y.mean(0))/test_y.std(0)

XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
                                                  y=test_y, time=t)

mse_test, _ = evaluate_model(trained_model,XTest,YTest,t_train=t_test,vis=1)