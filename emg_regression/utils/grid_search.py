#!/usr/bin/env python
import numpy as np
import pickle
import itertools
import torch


def grid_search(x,y,input_grid, 
                basisFunc_grid, 
                numKernels_grid,
                batchSize_grid, lr_grid, nEpochs_grid,
                trainingRatio,device):
    
    # Create a parameter generator to iterate through all the combinations of hyperparameters
    param_generator = iter(itertools.product(input_grid, 
                            basisFunc_grid, numKernels_grid,
                            nEpochs_grid, batchSize_grid, lr_grid))

    # Compute number of hyper-parameters combinations
    params = [input_grid, basisFunc_grid, numKernels_grid, nEpochs_grid, batchSize_grid, lr_grid]
    nb_params = len(params)
    nb_combinations = len(input_grid) * len(basisFunc_grid) * len(numKernels_grid) * len(nEpochs_grid) * len(batchSize_grid) * len(lr_grid) 
    print('Nb combinations:',nb_combinations)
    count = 0    # Initialize combinations' counter to zero

    # Create a grid to store hyperparameters, regression performances
    nb_res = 3   # train and test mse
    results_grid = np.zeros((nb_combinations, nb_params+nb_res),dtype=object)

    # other params
    normalize_input = True
    loss_fn = nn.MSELoss()
    seed = 0
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    debug = False

    while True:
        try:
            # Extract hyperparameters' combination
            param_list = next(param_generator)
            [input_name, basis_func, num_kernels, n_epochs, batch_size, lr] = param_list

            input_ = get_input(input_name,x)
            dim_input, dim_output = np.shape(input_)[1], np.shape(y)[1]
            input_norm = ((input_ - input_.mean(axis=0))/input_.std(axis=0) if normalize_input else input_)

            # Initialize input and output data
            X = torch.from_numpy(input_norm).float().to(device)
            Y = torch.from_numpy(y).float().to(device)
            XTrain,XTest,YTrain,YTest = train_test_split(X,Y,train_size=trainingRatio)

            # define and train model on 75% of data
            model = RBFNet(dim_input,dim_output,num_kernels,basis_func).to(device)
            loss_epochs, loss_batches = model.train(XTrain, YTrain, n_epochs, batch_size, lr, loss_fn,debug)

            train_error = F.mse_loss(model(XTrain), YTrain).cpu().detach().numpy()
            test_error  = F.mse_loss(model(XTest),  YTest).cpu().detach().numpy()
            print("Training err: ", train_error, "| Testing err: ", test_error) 

            # Compute number of parameters in the model
            total_nb_parameters = 0
            for p in model.parameters():
                total_nb_parameters += p.nelement()

            # Fill results' grid
            results_grid[count, 0] = input_name
            results_grid[count, 1] = basis_func
            results_grid[count, 2] = num_kernels
            results_grid[count, 3] = n_epochs
            results_grid[count, 4] = batch_size
            results_grid[count, 5] = lr
            results_grid[count, 6] = float(train_error)
            results_grid[count, 7] = float(test_error)
            results_grid[count, 8] = int(total_nb_parameters)
            
            count += 1
            print('Combinations tested: {:d}/{:d}'.format(count, nb_combinations))

        except StopIteration:
            return results_grid