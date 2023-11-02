#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle, datetime
from emg_regression.approximators.lstm import LSTM
from sklearn.metrics import mean_squared_error

def compute_loss(predicted, target_data):
    with torch.no_grad():
        return F.mse_loss(predicted, target_data)

def predict(model,X,Y):
    YPred = model(X)
    mse = compute_loss(YPred,Y).item()
    return YPred, mse

def evaluate_model(model,XTrain,YTrain,vis=None,t_train=None,XTest=None,YTest=None,t_test=None):
    YTrain_pred, mse_train = predict(model,XTrain,YTrain)
    # ytrain      = YTrain.detach().numpy()
    # ytrain_pred = YTrain_pred.detach().numpy()
    ytrain      = YTrain.detach().cpu().numpy()
    ytrain_pred = YTrain_pred.detach().cpu().numpy()
    t_train = np.arange(0,len(ytrain)*0.01,0.01) if t_train is None else t_train
    m, mse_test = 1, []
    output_dim = ytrain.shape[1]
    if XTest is not None:
        YTest_pred, mse_test = predict(model,XTest,YTest)
        # ytest      = YTest.detach().numpy()
        # ytest_pred = YTest_pred.detach().numpy()
        ytest      = YTest.detach().cpu().numpy()
        ytest_pred = YTest_pred.detach().cpu().numpy()
        t_test = np.arange(0,len(ytest)*0.01,0.01) if t_test is None else t_test
        m = 2
    if vis:
        fig, axes = plt.subplots(output_dim,m,figsize=(8,5))
        for j in range(output_dim):
            ax = axes[j, 0] if m > 1 else axes if output_dim == 1 else axes[j]
            ax.plot(t_train,ytrain[:,j],label='True')
            ax.plot(t_train,ytrain_pred[:,j], color='r',label='Predicted')
            ax.set_ylabel(r'$y_{}$'.format(j+1))
            ax.grid()
            if j == output_dim -1: ax.set_xlabel('Time [s]')
            if j == 0:
                ax.legend()
                ax.set_title(f'Training data (MSE = {mse_train:.5f})')
            if XTest is not None:
                ax = axes[j,1]
                ax.plot(t_test,ytest[:,j],label='True')
                ax.plot(t_test,ytest_pred[:,j], color='r', label='Predicted')
                ax.set_ylabel(r'$y_{}$'.format(j+1))
                ax.grid()
                axes[-1,1].set_xlabel('Time [s]')
                axes[0,1].set_title(f'Testing data (MSE = {mse_test:.5f})')

        plt.tight_layout()
        plt.show()
    return mse_train, mse_test


def plot_regression(ytrain, ytrain_pred, t_train,  ytest=None, ytest_pred=None,t_test=None):
    output_dim = ytrain.shape[1]
    mse_train = mean_squared_error(ytrain_pred,ytrain)
    m, mse_test = 1, []
    if ytest is not None:
      mse_test = mean_squared_error(ytest_pred,ytest)
      m = 2

    fig, axes = plt.subplots(output_dim,m,figsize=(8,5))
    for j in range(output_dim):
        ax = axes[j, 0] if m > 1 else axes if output_dim == 1 else axes[j]
        ax.plot(t_train,ytrain[:,j],label='True')
        ax.plot(t_train,ytrain_pred[:,j], color='r',label='Predicted')
        ax.set_ylabel(r'$y_{}$'.format(j+1))
        ax.grid()
        if j == output_dim -1: ax.set_xlabel('Time [s]')
        if j == 0:
            ax.legend()
            ax.set_title(f'Training data (MSE = {mse_train:.5f})')
        if ytest is not None:
            ax = axes[j,1]
            ax.plot(t_test,ytest[:,j],label='True')
            ax.plot(t_test,ytest_pred[:,j], color='r', label='Predicted')
            ax.set_ylabel(r'$y_{}$'.format(j+1))
            ax.grid()
            axes[-1,1].set_xlabel('Time [s]')
            axes[0,1].set_title(f'Testing data (MSE = {mse_test:.5f})')
    plt.tight_layout()
    plt.show()

    return mse_train, mse_test


def split_train_test(x,y,train_ratio,t=None):
    
    NTrain = int(len(x) * train_ratio)
    train_x = x[:NTrain,:]
    train_y = y[:NTrain,:]
    t_train = t[:NTrain] if t is not None else []

    test_x = x[NTrain:,:] if train_ratio != 1 else []
    test_y = y[NTrain:,:] if train_ratio != 1 else []
    t_test  = t[NTrain:] if t is not None else []
    
    return train_x, test_x, train_y, test_y, t_train, t_test


def save_model(path, model, params):
    time_  = datetime.datetime.now().strftime("_%H_%M") 
    model_name, params_name = '/lstm'+time_, '/params'+time_
    model_path = path + model_name +'.pt'
    params_path= path + params_name+'.sav'
    pickle.dump(params, open(params_path, 'wb'))  
    torch.save(model.state_dict(), model_path)
    print('MODEL SAVED to: ' + model_path)


def load_model(params_path,model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load params
    params = pickle.load(open(params_path,'rb'))
    dim_input      = params["dim_input"]
    dim_output     = params["dim_output"]
    dim_pre_output = params["dim_pre_output"]
    nb_layers      = params["nb_layers"]
    # bidirectional  = params["bidirectional"]
    bidirectional  = False

    # Load model
    model = LSTM(dim_input, dim_pre_output, nb_layers,
                dim_pre_output,dim_output,bidirectional).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()


# Select input and output
def get_input_output(train_x,train_y,option):
    if option == 1:
        input  = train_x # u1,u2
        output = train_y[:,:2] # theta, phi
        print("Input: u1(k), u2(k) -> Output: theta(k), phi(k)")

    if option == 2:
        input  = np.append(train_y[:-1,:2], train_x[:-1,:],axis=1) # theta(k), phi(k), u1(k), u2(k)
        output = train_y[1:,:2] # theta(k+1), phi(k+1)
        print("Input: theta(k), phi(k), u1(k), u2(k) -> Output: theta(k+1), phi(k+1) ")

    if option == 3: # Option 3: (works with any network)
        input  = np.append(train_y[:-1,:], train_x[:-1,:],axis=1) # theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k)
        output = train_y[1:,:2] # theta(k+1), phi(k+1)
        print("Input: theta(k), phi(k), theta_dot(k), phi_dot(k), u1(k), u2(k) -> Output: theta(k+1), phi(k+1)")
    return input, output


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0,axis=0),axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)
