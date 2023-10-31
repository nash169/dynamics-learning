#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.trainer import Trainer
from emg_regression.utils.model_tools import evaluate_model, predict, plot_regression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train data
data_x = np.load('data/train_x.npy') # x: u_theta, u_phi
data_y = np.load('data/train_y.npy') # y: theta, phi, theta_dot, phi_dot

# Option 1: input: u1,u2, output: theta, phi
# Option 2: input: theta(k), phi(k), u1(k), u2(k), output: theta(k+1), phi(k+1)

u = data_x
y = data_y[:,:2]

# fig = plt.figure()
# fig.add_subplot(211).plot(u)
# fig.add_subplot(212).plot(y)
# plt.show()

dt = 0.01             # 100 Hz , 100 samples/ second
t = np.arange(0,len(u)*dt,dt)

# create the dataset with windows:
time_window = 1 #seconds
window_size = int(time_window/dt)
print(f'Time-window = {time_window} s, nb_samples = {window_size}')

x = data_x
y = data_y[:,:2] #positions

# Normalize input
x_mu, x_std = x.mean(0), x.std(0)
y_mu, y_std = y.mean(0), y.std(0)

x = (x - x_mu)/x_std
y = (y - y_mu)/y_std

X = torch.empty(1, window_size, x.shape[1]).float().to(device)
Y = torch.empty(1, window_size, y.shape[1]).float().to(device)
x_ = torch.from_numpy(x) if isinstance(x,np.ndarray) else x
y_ = torch.from_numpy(y.copy()) if isinstance(y,np.ndarray) else y

# after must ignore the first value! It's a zero from initialization
for k in range(window_size-1,len(x)):
    x_tw = x_[k-window_size+1:k+1,:].unsqueeze(0)
    y_tw = y_[k-window_size+1:k+1,:].unsqueeze(0)
    X = torch.cat((X,x_tw))
    Y = torch.cat((Y,y_tw))

X, Y = X[1:].float(), Y[1:].float()

# FEED TIME-WINDOWS, PREDICT VALUE!
Xin, Yin = X[:-1], Y[:-1]
Yout = y_[window_size:].to(device).float()
t_   = t[window_size:]

input_  = torch.cat((Xin,Yin), dim=2)
output_ = Yout

# DS-LSTM
dim_input  = input_.shape[-1]
dim_output = output_.shape[-1]
num_layers = 2
dim_hidden = 20
dim_pre_output = 20
bidirectional = False

# train
num_epochs = 200
learning_rate = 1e-2
weight_decay  = 1e-5
training_ratio = 1
mini_batch_size = 50

# Model
approximator = LSTM(dim_input, dim_hidden, num_layers,
                    dim_pre_output, dim_output, bidirectional).to(device)

# Train
trainer = Trainer(model=approximator, input=input_, target=output_)
trainer.options(normalize=False,
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
mse_train, _ = evaluate_model(trained_model,input_,output_,vis=1)



# Try now to predict it iteratively with sequence of "real-time"
x = data_x
y = data_y[:,:2] #positions
t = np.arange(0,len(x)*dt,dt)

model = trained_model

train_input  = np.hstack((x[:-1,:],y[:-1,:]))
train_output = (y - y.mean(0))/y.std(0)

mu, std = train_input.mean(0), train_input.std(0)
Ypred_train = torch.empty(window_size,y.shape[1]).float().to(device) #must start with zeros

for k in range(window_size-1,len(x)):
    x_tw = train_input[k-window_size+1:k+1,:]

    # Normalize input
    x_tw = (x_tw - mu)/std
    X_tw = torch.from_numpy(x_tw).unsqueeze(0).float()
    
    ypred_k = model(X_tw)
    Ypred_train = torch.cat((Ypred_train,ypred_k))

ypred_train = Ypred_train[1:].detach().numpy()

# numpy arrays
ytrue = train_output[window_size:]
ypred = ypred_train[window_size:]
tpred = t[window_size:]
mse   = plot_regression(ytrue,ypred,tpred)

# It's working well
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')








# Now the ultimate test:

# Evaluation 



""" This method might work!! Compare with input emg"""

# Testing data
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')

# process testing data but predictions in "real-time" like structure
xtest = test_x
ytest = test_y[:,:2] #positions

# Normalize
xtest_mu, xtest_std = xtest.mean(0), xtest.std(0)
ytest_mu, ytest_std = ytest.mean(0), ytest.std(0)

# Train in/out normalization: numpy x_mu, x_std, y_mu, y_std
Xtrain_mu, Xtrain_std = torch.from_numpy(x_mu).to(device), torch.from_numpy(x_std).to(device)
Ytrain_mu, Ytrain_std = torch.from_numpy(y_mu).to(device), torch.from_numpy(y_std).to(device)

# Test
Xtest_mu,  Xtest_std  = torch.from_numpy(xtest_mu).to(device), torch.from_numpy(xtest_std).to(device)
Ytest_mu,  Ytest_std  = torch.from_numpy(ytest_mu).to(device), torch.from_numpy(ytest_std).to(device)

Ypred_test = torch.empty(window_size,ytest.shape[1]).float().to(device) #must start with zeros

xin = xtest
xin = x

for k in range(window_size-1,len(xin)):
    x_tw = xin[k-window_size+1:k+1,:]

    # Normalize input
    # x_tw = (x_tw - xtest_mu)/xtest_std  
    x_tw = (x_tw - x_mu)/x_std  
    X_tw = torch.from_numpy(x_tw).unsqueeze(0).float()

    # Normalize output
    Y_tw = Ypred_test[-window_size:,:].float()
    # Y_tw = Y_tw.sub_(Ytrain_mu).div_(Ytrain_std).float()

    # Compose input with previous predictions
    input_tw = torch.cat((X_tw,Y_tw.unsqueeze(0)),dim=2)
    ypred_k = model(input_tw)
    Ypred_test = torch.cat((Ypred_test,ypred_k))

ypred_test = Ypred_test[window_size:].detach().numpy()
t_test = np.arange(0,len(ytrain)*dt,dt)
ytest = ytrain
# t_test = np.arange(0,len(ytest)*dt,dt)

nb_outputs = ytest.shape[1]
fig, axes = plt.subplots(nb_outputs,1,figsize=(8,5))
for j in range(nb_outputs):
    ax = axes[j]
    ax.plot(t_test,ytest[:,j],label='True')
    ax.plot(t_test,ypred_test[1:,j], color='r',label='Predicted')
    ax.set_ylabel(r'$y_{}$'.format(j+1))
    # ax.set_xticks(np.arange(0,len(ytest)*dt,1))
    ax.grid()
    if j == nb_outputs -1: 
        ax.set_xlabel('Time [s]')
    if j == 0:
        ax.legend()
        ax.set_title(f'Training data (MSE = {mse_train:.5f})')
plt.tight_layout()
plt.show()






t = np.arange(0,len(u)*dt,dt)

# create the dataset with windows:
time_window = 1 #seconds
window_size = int(time_window/dt)
print(f'Time-window = {time_window} s, nb_samples = {window_size}')

x = data_x
y = data_y[:,:2] #positions

X = torch.empty(1, window_size, x.shape[1]).float().to(device)
Y = torch.empty(1, window_size, y.shape[1]).float().to(device)
x_ = torch.from_numpy(x) if isinstance(x,np.ndarray) else x
y_ = torch.from_numpy(y.copy()) if isinstance(y,np.ndarray) else y

# after must ignore the first value! It's a zero from initialization
for k in range(window_size-1,len(x)):
    x_tw = x_[k-window_size+1:k+1,:].unsqueeze(0)
    y_tw = y_[k-window_size+1:k+1,:].unsqueeze(0)
    X = torch.cat((X,x_tw))
    Y = torch.cat((Y,y_tw))

X, Y = X[1:].float(), Y[1:].float()

# FEED TIME-WINDOWS, PREDICT VALUE!
Xin, Yin = X[:-1], Y[:-1]
Yout = y_[window_size:].to(device).float()
t_   = t[window_size:]

input_  = torch.cat((Xin,Yin), dim=2)
output_ = Yout






# input vector
# Option 2: input: [theta(k), phi(k), u1(k), u2(k)], output: theta(k+1), phi(k+1)
# Get the emg signal
# Utest = torch.from_numpy(train_x[:,2:])
# Xtest = XTrain[:,:,:2]
# Utest = XTrain[:,:,2:]

# Ypred = torch.zeros((window_size, dim_output))
# # Ypred = YTrain[:window_size,:]
# Xtest = torch.zeros((1, dim_input))

# # must add the offset and fix Xtest
# for k in range(0,len(Utest)-window_size-1):
#     # input = [theta_k, phi_k, u1_k, u2_k] 
#     Utest_window = Utest[k+window_size+1]
#     Ypred_window = YTrain[k:k+window_size,:] #Ypred[-window_size:] 

#     XTest_k = torch.hstack((Ypred_window,Utest_window)).unsqueeze(0).float()
#     with torch.no_grad():
#         ypred_k = model(XTest_k.to(device))
#     Ypred = torch.vstack((Ypred,ypred_k))
#     Xtest = torch.vstack((Xtest,XTest_k[0,-1,:]))

# Ypred = Ypred[window_size-1:,:]
# ytrue = YTrain[:,:2].detach().numpy()
# ypred =  Ypred.detach().numpy()

# m = window_size-offset +2
# # m = 0
# tt1,tt2 = t_train,t_train[m:]
# fig, ax = plt.subplots(2,1)
# for j in range(2):
#     ax[j].plot(ytrue[m:,j],'r',label='True')
#     ax[j].plot(ypred[:,j],'b',label='Predicted')
# ax[0].legend()
# plt.show()

# compute_loss(Ypred, YTrain[window_size:,:2])


# #########

# # Trainer
# loss_fun  = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(approximator.parameters(), lr=learning_rate, weight_decay=weight_decay)
# loss_log = []

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     pred = approximator(train_x)
#     loss = loss_fun(pred,train_y)
#     loss_log.append(loss.item())
#     loss.backward()
#     optimizer.step()

# loss_log = np.array(loss_log)
# # Train
# trainer = Trainer(model=approximator, input=XTrain, target=YTrain)
# trainer.options(normalize=True,
#                 epochs=num_epochs,
#                 batch=mini_batch_size,
#                 shuffle=False,
#                 record_loss=True)

# trainer.optimizer = torch.optim.Adam(approximator.parameters(), 
#                                      lr=learning_rate, 
#                                      weight_decay=weight_decay)

# loss = trainer.train()
# trained_model = trainer.model
# plt.plot(trainer.losses)
# plt.show()

# mse_train, _ = evaluate_model(trained_model,XTrain,YTrain,vis=1)

# # Test
# # Testing data
# test_x = np.load('data/test_x.npy')
# test_y = np.load('data/test_y.npy')

# test_x, test_y = get_input_output(test_x,test_y,option=1)
# # test_x = (test_x - train_x.mean(0))/train_x.std(0)
# # test_y = (test_y - test_y.mean(0))/test_y.std(0)

# XTest, YTest, t_test = approximator.process_input(test_x, window_size, offset,
#                                                   y=test_y, time=t)

# mse_test, _ = evaluate_model(trained_model,XTest,YTest,t_train=t_test,vis=1)