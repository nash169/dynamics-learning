#!/usr/bin/env python

import torch
import os
import numpy as np

class Trainer:
    __options__ = ['epochs', 'batch', 'normalize',
                   'shuffle', 'record_loss', 'print_loss', 'clip_grad', 'load_model']

    def __init__(self, model, input, target):
        # Set the model
        self.model = model

        # Set the input
        self.input = input

        # Set the target
        self.target = target

        # Set deault optimizer
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
            load_model=None)

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
            self.options_['batch'] = self.input.size(0)

        # Normalize dataset
        if self.options_['normalize']:
            mu, std = self.input.mean(0), self.input.std(0)
            self.input.sub_(mu).div_(std)

        # Activate grad
        self.input.requires_grad = True

        # Create dataset
        torch_dataset = torch.utils.data.TensorDataset(self.input, self.target)

        # Create loader
        loader = torch.utils.data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.options_['batch'],
            shuffle=self.options_['shuffle'],
            num_workers=0
        )

        # Open file
        if self.options_["record_loss"]:
            loss_log = np.empty((0, 3))
            self.losses = np.array([])

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100, threshold=1e-2,
                                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)

        try:
            # start training
            for epoch in range(self.options_['epochs']):
                epoch_loss = np.array([])
                for iter, (batch_x, batch_y) in enumerate(loader):  # for each training step
                    b_x = torch.autograd.Variable(batch_x, requires_grad=True)
                    b_y = torch.autograd.Variable(batch_y)

                    # clear gradients for next train
                    self.optimizer.zero_grad()

                    # input x and predict based on x
                    prediction = self.model(b_x)

                    # must be (1. nn output, 2. target)
                    loss = self.loss(prediction, b_y)

                    # Record loss (EPOCH,ITER,LOSS)
                    if self.options_["record_loss"]:
                        epoch_loss = np.append(epoch_loss,loss.item())
                        loss_log = np.append(loss_log, np.array(
                            [epoch, iter, loss.item()])[np.newaxis, :], axis=0)

                    # backpropagation, compute gradients
                    loss.backward()

                    # Clip grad if requested
                    if self.options_["clip_grad"] is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.options_["clip_grad"])

                    # apply gradients
                    self.optimizer.step()

                    # step scheduler
                    scheduler.step(loss)
                    
                if self.options_["record_loss"]:
                    self.losses = np.append(self.losses, np.mean(epoch_loss))
                
                # Print loss
                if self.options_["print_loss"]:
                    print("EPOCH: ", epoch, " |  LOSS: ", self.losses[-1])

        except KeyboardInterrupt:
            return self.losses

        # Close file
        if self.options_["record_loss"]:
            return self.losses

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
