#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle, datetime

class LSTM(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_pre_output_layer, dim_output, bidirectional):
        super(LSTM, self).__init__()

        self.dim_input = dim_input

        self.dim_output = dim_output

        self.layer_dim = num_layers

        self.hidden_dim = dim_recurrent

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=dim_input, 
                            hidden_size=dim_recurrent,
                            num_layers=num_layers,
                            bidirectional=bidirectional, 
                            batch_first=True)
        
        self.D = (2 if bidirectional else 1)

        self.fc_o2y = nn.Linear(in_features=dim_recurrent*self.D, out_features=dim_pre_output_layer)

        self.fc_y2c = nn.Linear(in_features=dim_pre_output_layer, out_features=dim_output)

    def forward(self, X):

        batch_size = X.size(0)
        # Initialize hidden state vector and cell state
        h0 = torch.zeros(self.D*self.layer_dim, batch_size, self.hidden_dim).to(X.device)
        c0 = torch.zeros(self.D*self.layer_dim, batch_size, self.hidden_dim).to(X.device)

        # output, _ = self.lstm(X, (h0.detach(), c0.detach()))
        output, _ = self.lstm(X, (h0, c0))
        output = self.fc_o2y(output[:, -1, :])
        output = F.relu(output)
        output = self.fc_y2c(output)

        return output
    
    
    def process_input(self, x, window_size, offset, y=None, time=None):
        """
        Reshape data in time-windows displaced by an offset for time-series label prediction 
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.empty(1, window_size, self.dim_input).float().to(device)
        Y = torch.empty(1, self.dim_output).float().to(device)

        X0 = torch.from_numpy(x) if isinstance(x,np.ndarray) else x
        Y0 = torch.from_numpy(y.copy()) if y is not None and isinstance(y, np.ndarray) else y
        X0 = X0.to(device)
        Y0 = Y0.to(device)

        t  = np.array([])

        for k in range(0, X0.size(0)-window_size+1, offset):
            x = X0[k:k+window_size, :].unsqueeze(0)
            X = torch.cat((X, x))
            if y is not None:
                y = Y0[k+window_size-1].view(1,-1)
                Y = torch.cat((Y, y))
            if time is not None:
                t = np.append(t,time[k+window_size-1])

        return X[1:].float(), Y[1:,:].float(), t
    
    def plot_loss(self,loss):
        plt.subplots(figsize=(5,4))
        plt.plot(loss)
        plt.ylabel('MSE loss')
        plt.xlabel('Epochs')
        plt.tight_layout()
        plt.show()


    def train_model(self,model,x,y,training_ratio,window_size,offset,
              num_epochs, batch_size, learning_rate, weight_decay, t):

        self.training_ratio = training_ratio
        self.NTrain = int(len(x)*self.training_ratio)
        t_train, t_test  = t[:self.NTrain],   t[self.NTrain:]
        train_x, train_y = x[:self.NTrain,:], y[:self.NTrain,:]
        if self.training_ratio < 1:
            test_x, test_y = x[self.NTrain:,:], y[self.NTrain:,:]

        mu, std = train_x.mean(0), train_x.std(0)
        train_x = (train_x-mu)/std

        self.params = {"mu": mu, "std": std, "maxs":train_x.max(0),
                    "window_size": window_size, "offset": offset,
                    "dim_input": self.dim_input, "dim_output": self.dim_output, 
                    "dim_pre_output": self.hidden_dim, "nb_layers": self.layer_dim,
                    "bidirectional": self.bidirectional,
                    "mini_batch_size": batch_size, "learning_rate": learning_rate, 
                    "weight_decay": weight_decay, "nb_epochs": num_epochs}        
        
        self.XTrain, self.YTrain, self.t_train = self.process_input(train_x,train_y,window_size,offset,t_train)
        if self.training_ratio < 1:
            self.XTest,  self.YTest,  self.t_test  = self.process_input(test_x,test_y,window_size,offset,t_test)
            print('\nTrain:',self.XTrain.shape,', Test:',self.YTrain.shape,'\n')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        try:
            dataset    = torch.utils.data.TensorDataset(self.XTrain, self.YTrain)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # larger weight -> stronger regularization
            loss_fun = torch.nn.MSELoss()
            loss_log = []

            # Loop over the epochs
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                model.train()  
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = loss_fun(pred, targets)
                    loss.backward()   # Back-propagate gradient
                    optimizer.step()  # Update parameters
                    epoch_loss += loss.item()

                loss_log.append(epoch_loss / len(dataloader))  # Average loss per batch
                print('Epoch:', epoch, '| Loss:', loss_log[-1])

        except KeyboardInterrupt:
            print("Training interrupted. Returning current model and loss.")
            torch.cuda.empty_cache()
            self.evaluate_model(model)
            return model, loss_log  

        torch.cuda.empty_cache()
        self.evaluate_model(model)
        return model, loss_log

    def plot_prediction(self):

        ytrain = np.array(self.YTrain)
        ypred_train = self.Ypred_train.detach().numpy()
        m = 1
        if self.training_ratio < 1:
            m = 2
            ytest  = np.array(self.YTest)
            ypred_test  = self.Ypred_test.detach().numpy()

        fig, axes = plt.subplots(self.dim_output,m,figsize=(8,5))
        for j in range(self.dim_output):
            ax = axes[j,0] if m >1 else axes[j]
            ax.plot(self.t_train,ytrain[:,j],label='True')
            ax.plot(self.t_train,ypred_train[:,j], color='r',label='Predicted')
            ax.set_ylabel(r'$y_{}$'.format(j+1))
            if j == self.dim_output -1: ax.set_xlabel('Time [s]')
            if j == 0:
                ax.legend()
                ax.set_title(f'Training data (MSE = {self.train_error:.5f})')
            if self.training_ratio < 1:
                ax = axes[j,1]
                ax.plot(self.t_test,ytest[:,j],label='True')
                ax.plot(self.t_test,ypred_test[:,j], color='r', label='Predicted')
                ax.set_ylabel(r'$y_{}$'.format(j+1))
                axes[-1,1].set_xlabel('Time [s]')
                axes[0,1].set_title(f'Testing data (MSE = {self.test_error:.5f})')
        plt.tight_layout()
        plt.show()

    def evaluate_model(self,model):
        self.Ypred_train = model(self.XTrain)
        self.train_error = self.compute_loss(self.Ypred_train, self.YTrain).item()
        print("Training error: ", self.train_error)
        if self.training_ratio < 1:
            self.Ypred_test  = model(self.XTest)
            self.test_error  = self.compute_loss(self.Ypred_test,  self.YTest).item()
            print("Testing error: ",  self.test_error)

    def compute_loss(self, predicted, target_data):
        with torch.no_grad():
            return F.mse_loss(predicted, target_data)

    def save_model(self, path, model):
        time_  = datetime.datetime.now().strftime("_%H_%M") 
        model_name, params_name = '/lstm-1'+time_, '/params-1'+time_
        model_path = path + model_name +'.pt'
        params_path= path + params_name+'.sav'
        pickle.dump(self.params, open(params_path, 'wb'))  
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

    return model, params

def process_input2(x,window_size,offset,time):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X0 = torch.from_numpy(x) if isinstance(x,np.ndarray) else x
    X = torch.empty(1, window_size, x.shape[1]).float().to(device)
    t = np.array([])
    for k in range(0, X0.size(0)-window_size+1, offset):
        x = X0[k:k+window_size, :].unsqueeze(0)
        t = np.append(t,time[k+window_size-1])
        X = torch.cat((X, x))
    return X[1:].float(), t
