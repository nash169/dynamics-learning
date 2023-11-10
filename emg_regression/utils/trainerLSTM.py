#!/usr/bin/env python

import torch
import os
import numpy as np

class Trainer:
    __options__ = ['epochs', 'batch', 'normalize_input', 'normalize_output',
                   'shuffle', 'record_loss', 'print_loss', 'clip_grad', 'load_model','stateful_train']

    def __init__(self, model, input, target):
        # Set the model
        self.model = model

        # Get number of trajectories
        self.nb_trajectories = input.size(0) if input.dim() == 4 else 1

        # Set the input
        self.input = input.unsqueeze(0) if self.nb_trajectories == 1 else input

        # Set the target
        self.target = target.unsqueeze(0) if self.nb_trajectories == 1 else target

        # Set default optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-4,  weight_decay=1e-8)

        # Set default loss function
        self.loss = torch.nn.MSELoss()

        # Set default options
        self.options_ = dict(
            epochs=5,
            batch=None,
            normalize=False,
            shuffle=True,
            record_loss=False,
            print_loss=True,
            clip_grad=None,
            load_model=None,
            stateful_train=True
            )


    def options(self, **kwargs):
        for _, option in enumerate(kwargs):
            if option in self.__options__:
                self.options_[option] = kwargs.get(option)
            else:
                print("Warning: option not found -", option)


    def train(self):
        # Load model if requested
        if self.options_['load_model'] is not None:
            self.load(self.options_['load_model'])

        # Set batch to dataset size as default
        if self.options_['batch'] is None:
            self.options_['batch'] = self.input.size(0) if self.nb_trajectories==1 else self.input.size(1)

        # Activate grad
        # self.input.requires_grad = True

        # Open file
        if self.options_["record_loss"]:
            self.epoch_losses = np.array([])

        # Scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100, threshold=1e-2,
        #                                                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)

        try:
            # start training
            for epoch in range(self.options_['epochs']):
                # for each trajectory
                traj_losses = np.array([])
                for traj in range(self.nb_trajectories):

                    # Create loader for batch training #TODO personalize to load by traj and then batch
                    torch_dataset = torch.utils.data.TensorDataset(self.input[traj], self.target[traj])
                    loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                                         batch_size=self.options_['batch'],
                                                         shuffle=self.options_['shuffle'],
                                                         drop_last=False, #otherwise gives error for h0,c0 shape
                                                         num_workers=0)

                    # Initialize hidden states
                    (h0, c0) = self.model.initialize_states(batch_size=self.options_['batch'])

                    batch_losses = []
                    for batch_x, batch_y in loader:  # for each training step
                        b_x = torch.autograd.Variable(batch_x)
                        b_y = torch.autograd.Variable(batch_y)

                        if len(b_x) != self.options_['batch']:
                            b_x = torch.cat((b_x, self.pad_x * b_x[-1,-1,:]))
                            b_y = torch.cat((b_y, self.pad_y * b_y[-1,:]))

                        # clear gradients for next train
                        self.optimizer.zero_grad()

                        # input x and predict based on x
                        if self.options_['stateful_train']:
                            prediction, (h0, c0) = self.model(b_x,(h0, c0)) # stateful training
                        else:
                            prediction,_ = self.model(b_x) # stateless training

                        # must be (1. nn output, 2. target)
                        loss = self.loss(prediction, b_y)

                        # backpropagation, compute gradients
                        loss.backward()

                        # to treat each batch individually ??? else we maintain continuity between batches
                        h0.detach(), c0.detach()  

                        # Clip grad if requested
                        if self.options_["clip_grad"] is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.options_["clip_grad"])
                
                        # Loss per batch
                        if self.options_["record_loss"]:
                            batch_losses.append(loss.item())

                        # apply gradients
                        self.optimizer.step()


                        # step scheduler
                        # scheduler.step(loss)

                    # Loss per trajectory
                    if self.options_["record_loss"]:
                        ave_traj_loss = np.array(batch_losses).mean()
                        traj_losses = np.append(traj_losses, ave_traj_loss)
                        # print("EPOCH:", epoch, "| TRAJ:",traj, " | LOSS:", ave_traj_loss)

                # Record loss  per epoch (EPOCH,ITER,LOSS)
                if self.options_["record_loss"]:
                    ave_epoch_loss = traj_losses.mean()
                    self.epoch_losses = np.append(self.epoch_losses,ave_epoch_loss)

                # Print loss
                if self.options_["print_loss"]:
                    print("EPOCH: ", epoch, " |  LOSS: ", ave_epoch_loss)

        except KeyboardInterrupt:
            return self.epoch_losses

        # Close file
        if self.options_["record_loss"]:
            return self.epoch_losses

        return self.epoch_losses

    # Model
    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, value):
        self.model_ = value

    # Input
    @property
    def input(self):
        return self.input_

    @input.setter
    def input(self, value):
        self.input_ = value

    # Target
    @property
    def target(self):
        return self.target_

    @target.setter
    def target(self, value):
        self.target_ = value

    # Optimizer
    @property
    def optimizer(self):
        return self.optimizer_

    @optimizer.setter
    def optimizer(self, value):
        self.optimizer_ = value

    # Loss function
    @property
    def loss(self):
        return self.loss_

    @loss.setter
    def loss(self, value):
        self.loss_ = value

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def pad_size(self):
        return self.options_['batch'] - (self.input.size(-3) % self.options_['batch'])

    @property
    def pad_x(self):
        return torch.ones(self.pad_size,self.input.size(-2),self.input.size(-1), device=self.device)

    @property
    def pad_y(self):
        return torch.ones(self.pad_size,self.target.size(-1), device=self.device)