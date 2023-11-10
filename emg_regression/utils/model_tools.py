#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle, datetime
from emg_regression.approximators.lstm import LSTM
from sklearn.metrics import mean_squared_error
import random

def compute_loss(predicted, target_data):
    with torch.no_grad():
        return F.mse_loss(predicted, target_data)

def predict(model,X,Y):
    with torch.no_grad():
        YPred,_ = model(X)
        mse = compute_loss(YPred,Y).item()
    return YPred, mse

def evaluate_model(model,XTrain,YTrain,vis=None,t_train=None,XTest=None,YTest=None,t_test=None,bias=None):
    # added inital offset and bias correction
    
    YTrain_pred, mse_train = predict(model,XTrain,YTrain)
    # ytrain      = YTrain.detach().numpy()
    # ytrain_pred = YTrain_pred.detach().numpy()
    ytrain      = YTrain.detach().cpu().numpy()
    ytrain_pred = YTrain_pred.detach().cpu().numpy() - bias
    offset = ytrain_pred[0,:] - ytrain[0,:]
    ytrain_pred = ytrain_pred - offset

    t_train = np.arange(0,len(ytrain)*0.01,0.01) if t_train is None else t_train
    m, mse_test = 1, []
    output_dim = ytrain.shape[1]
    bias = 0 if bias is None else bias
    if XTest is not None:
        YTest_pred, mse_test = predict(model,XTest,YTest)
        # ytest      = YTest.detach().numpy()
        # ytest_pred = YTest_pred.detach().numpy()
        ytest      = YTest.detach().cpu().numpy()
        ytest_pred = YTest_pred.detach().cpu().numpy() - bias
        offset = ytest_pred[0,:]  - ytest[0,:] 
        ytest_pred = ytest_pred - offset

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
    t_test = t[NTrain:] if train_ratio != 1 else []
    
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


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0,axis=0),axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Trajectory processing
def prepare_dataset(u_traj,x_traj,t,goal,imu_inputs,imu_outputs):
    """" Choose between pos, vel or pos_vel to use as the input and output 
        Goal: SI (system identification) or ForwardPred
    """
    if imu_inputs == 'pos':
        in_idxs = [0,1]
    if imu_inputs == 'vel':
        in_idxs = [2,3]
    if imu_inputs == 'pos_vel':
        in_idxs = [0,1,2,3]   

    if imu_outputs == 'pos':
        out_idxs = [0,1]
    if imu_outputs == 'vel':
        out_idxs = [2,3]
    if imu_outputs == 'pos_vel':
        out_idxs = [0,1,2,3]
    
    dim = len(x_traj.shape) #if 3, I got (nbTraj,n,m)
    if goal == 'SI': #system identification
        xin  = u_traj
        yout = x_traj[:,:,out_idxs] if dim == 3 else x_traj[:,out_idxs]
        t_ = t

    if goal == 'ForwardPred': #system identification
        if dim == 3:
            xin  = np.concatenate((u_traj[:,:-1,:], x_traj[:,:-1,in_idxs]), axis=2)
            yout = x_traj[:,1:,out_idxs]
        else:
            xin  = np.concatenate((u_traj[:-1,:], x_traj[:-1,in_idxs]), axis=1)
            yout = x_traj[1:,out_idxs]           
        t_ = t[1:]

    return xin, yout, t_


# Normalization
def normalize_dataset(x_train,y_train,
                      normalize_input,normalize_states,
                      input_norm,states_norm,
                      norms=None,
                      vis=None):

    input, target = x_train.reshape(-1,x_train.shape[-1]), y_train.reshape(-1,y_train.shape[-1])
    dim_u = 2
    dim_inputstates = input.shape[-1] - dim_u
    
    if norms is None:
        u_mu , u_std = input[:,:dim_u].mean(0), input[:,:dim_u].std(0)
        u_min, u_max = abs(input[:,:dim_u].min(0)),  abs(input[:,:dim_u].max(0))
        y_mu , y_std = target.mean(0),target.std(0)
        y_min, y_max = abs(target.min(0)), abs(target.max(0))

    else:
        u_mu , u_std, u_min, u_max, y_mu , y_std, y_min, y_max = norms

    input_n = input.copy()
    target_n = target.copy()
    
    if normalize_input:
        if input_norm == 'std':
            input_n[:,:dim_u] = (input[:,:dim_u] - u_mu)/u_std

        if input_norm == 'min-max':
            for j in range(0,dim_u):
                input_n[:,j] = np.where(input[:,j]>=0, input[:,j]/u_max[j], input[:,j]/u_min[j])

    if normalize_states:
        # normalize both in input and output
        if states_norm == 'std':
            input_n [:,dim_u:] = (input[:,dim_u:] - y_mu)/y_std
            target_n = (target - y_mu)/y_std

        if states_norm == 'min-max':
            for j in range(dim_u,dim_u+dim_inputstates):
                input_n[:,j] = np.where(input[:,j]>=0, input[:,j]/y_max[j-dim_u], input[:,j]/y_min[j-dim_u])
            
            for j in range(y_train.shape[-1]):
                target_n[:,j] = np.where(target[:,j]>=0, target[:,j]/y_max[j], target[:,j]/y_min[j])

    if vis:
        plt.plot(input_n[:,:])
        plt.plot(target_n[:,:])
        plt.show()    
    
    # reshape data
    input_n  = input_n.reshape(x_train.shape)
    target_n = target_n.reshape(y_train.shape)

    norms = [ u_mu , u_std, u_min, u_max, y_mu , y_std, y_min, y_max]

    return input_n, target_n, norms


def min_max_scaling(x,xmin,xmax):
    return torch.where(x>=0, x/xmax, x/xmin)  

def split_train_test_traj(xin,yout,nb_traj,training_ratio,shuffle=True):
    nb_train = int(nb_traj*training_ratio)
    nb_test  = nb_traj-nb_train
    traj_idxs = np.arange(nb_traj)
    if shuffle:
        random.Random(4).shuffle(traj_idxs)
    # nb_train = 35
    # nb_test = 15

    xin_train  = xin[traj_idxs[:nb_train],:,:]
    yout_train = yout[traj_idxs[:nb_train],:,:]
    xin_test   = xin[traj_idxs[nb_train:],:,:]
    yout_test  = yout[traj_idxs[nb_train:],:,:]

    return xin_train, yout_train, xin_test, yout_test


def plot_predicted_paths(axes,x_test,y_test,model,window_size,init_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim_input, dim_output = x_test.shape[2], y_test.shape[2]
    X_init = torch.zeros(init_samples,window_size,dim_input).to(device)
    Y_init = torch.zeros(init_samples,dim_output).to(device)
    Ypred_init, mse_offset = predict(model, X_init,Y_init)
    ytrue, ypred = Y_init.cpu().detach().numpy(), Ypred_init.cpu().detach().numpy()
    bias = (ypred - ytrue).mean(0)
    all_mse = []

    for i, ax in enumerate(axes.flat):
        input_  = x_test[i,:,:] # theta, phi, theta_dot, phi_dot
        target_ = y_test[i,:,:] # u = uc + ug

        XTest, YTest, T_test = model.process_input(input_, window_size, offset=1,
                                                    y=target_, time=None)

        XTest_ = torch.cat((X_init,XTest),dim=0)
        YTest_ = torch.cat((Y_init,YTest),dim=0)

        YTest_pred, mse_test = predict(model,XTest_,YTest_)
        ytest_true = YTest.cpu().detach().numpy()
        ytest_pred = YTest_pred[init_samples:].cpu().detach().numpy() - bias
        offset = ytest_pred[0] - ytest_true[0]
        ytest_pred = ytest_pred - offset
        all_mse.append(mse_test)

        ax.plot(ytest_true[:,0],ytest_true[:,1],alpha=1,color='k')
        ax.plot(ytest_pred[:,0],ytest_pred[:,1],alpha=1,color='grey')
        ax.scatter(ytest_true[0,0],ytest_true[0,1],  s=50,color='blue')
        ax.scatter(ytest_true[-1,0],ytest_true[-1,1],s=50,color='maroon')
        ax.scatter(ytest_pred[0,0],ytest_pred[0,1],  s=50,color='skyblue')
        ax.scatter(ytest_pred[-1,0],ytest_pred[-1,1],s=50,color='indianred')
        ax.set_title(f'MSE={mse_test:.5f}')

    ave_mse = np.array(all_mse).mean(0)
    plt.suptitle(f'Average Test MSE={ave_mse:.5f}', fontweight='bold')
    plt.tight_layout()
    plt.show()
    return ave_mse

def get_sliding_windows(input_tw_hat,window_size,device):
    input_dim = input_tw_hat.shape[1]
    Input_tw = torch.empty((1,window_size,input_dim)).to(device)
    for k in range(0,len(input_tw_hat),1):
        input_tw_k = input_tw_hat[k:k+window_size].unsqueeze(0)
        try: Input_tw = torch.cat((Input_tw,input_tw_k))
        except: pass
    return Input_tw[1:]

# Forward predictions
def forward_prediction(model,x,y,window_size,batch_size,correct_bias=False):
    # add different options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    u = x[:,:2]
    u_dim, y_dim = u.shape[1], y.shape[1]
    num_samples_batch = window_size + (batch_size -1)

    U = torch.from_numpy(u).float().to(device)
    Y = torch.from_numpy(y).float().to(device)
    input_tw_hat = torch.from_numpy(x[:num_samples_batch,:]).float().to(device)
    ypred = np.array([]).reshape(0,2)

    # input_tw_hat   = torch.zeros((num_samples_batch-1,u_dim+y_dim)).to(device)
    # input_tw_real  = torch.zeros((num_samples_batch-1,u_dim+y_dim)).to(device)
    # ypred          = np.zeros((1,y_dim))
    # ypred_real     = np.zeros((1,y_dim))
    # ypred_mem      = np.zeros((1,y_dim))
    # ypred_real_mem = np.zeros((1,y_dim))

    # X_init = torch.zeros((10,window_size,u_dim+y_dim)).to(device)
    # Y_init = model(X_init)
    # bias = (Y_init.cpu().detach().numpy()).mean(0)

    (h0, c0) = model.initialize_states(batch_size)
    # (h1, c1) = model.initialize_states(batch_size)
    bias = 0 
    with torch.no_grad():
        for i in range(num_samples_batch,len(u)):

            # create windows
            Input_tw_hat  = get_sliding_windows(input_tw_hat,window_size,device)

            # Model prediction trying different inputs and memory cell configurations
            # Ypred_hat, (h0,c0) = model(Input_tw_hat, (h0,c0))
            Ypred_hat, _ = model(Input_tw_hat)

            # append new prediction
            ypred_hat = Ypred_hat[-1,np.newaxis].cpu().detach().numpy()
            
            if correct_bias and i == num_samples_batch:
                bias = ypred_hat

            ypred_hat = ypred_hat - bias
            ypred = np.vstack((ypred,ypred_hat))          

            # Receive new sample and append previous prediction
            u_i = U[i,:].unsqueeze(0)
            input_i_hat  = torch.cat((u_i,Ypred_hat[-1,np.newaxis]),dim=1)
    
            # append new sample to past time window and slide window
            input_tw_hat  = torch.cat((input_tw_hat,input_i_hat))[-num_samples_batch:,:]
            
    fig, ax = plt.subplots(2,1)
    for m in range(2):
        ax[m].plot(y[num_samples_batch:,m],label='ytrue')
        ax[m].plot(ypred[:,m],label='ypred_in')
        ax[m].grid()
    ax[0].legend()
    plt.show()

    return ypred


def process_input_traj(x_train,y_train,model,window_size,len_traj,device):
    dim_input, dim_output  = x_train.shape[2], y_train.shape[2]

    # Create dataset
    X = torch.zeros(1,len_traj-window_size,window_size,dim_input).to(device)
    Y = torch.zeros(1,len_traj-window_size,dim_output).to(device)

    for k in range(len(x_train)):
        input_  = x_train[k,:,:] # theta, phi, theta_dot, phi_dot
        target_ = y_train[k,:,:] # u = uc + ug

        # Create time-windows for training
        XTrain, YTrain, T_train = model.process_input(input_, window_size, offset=1,
                                                    y=target_, time=None)
        
        X = torch.cat((X,XTrain.unsqueeze(0)))
        Y = torch.cat((Y,YTrain.unsqueeze(0)))

    return X[1:], Y[1:]