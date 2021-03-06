#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:59:34 2018

@author: Raul Sanchez-Vazquez
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data_utils

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split

class AutoEnc(nn.Module):
    '''
    Basic Auto-Encoder.

    Optimized via Mean Square Error Loss and Adam optimizer.

    It has the form:

    AutoEnc(
      (Encoder): Sequential(
        (0): INPUT_LAYER to H_DIM_LAYER
        (1): ReLU()
        (2): H_DIM_LAYERT to Z_DIM_LAYER
        (3): ReLU()
        (4): Dropout(p=0.0)
      )
      (Decoder): Sequential(
        (0): Z_DIM_LAYER to H_DIM_LAYET
        (1): ReLU()
        (2): H_DIM_LAYERT to INPUT_LAYER
      )
    )

    Parameters
    ----------
        mb_size : int
        z_dim : int
        h_dim : in
        lr : float
        train_size : float
        epochs : int
        p_dropout : float
        weight_decay : float
        allow_cuda : bool
        verbose : bool
        random_state : int
    '''
    def __init__(
        self,
        mb_size=32,
        z_dim=5,
        h_dim=128,
        lr=1e-3,
        train_size=1.,
        epochs=100,
        p_dropout=0.0,
        weight_decay=0,
        allow_cuda=False,
        verbose=False,
        random_state=None):

        super(AutoEnc, self).__init__()

        # General
        self.mb_size = mb_size
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.lr = lr
        self.train_size = train_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.p_dropout = p_dropout

        # Misc
        self.allow_cuda = allow_cuda
        self.verbose = verbose
        self.random_state = random_state

        if not self.random_state is None:
            torch.manual_seed(self.random_state)

        # Internal
        self.fitted = False
        if self.train_size < 1.:
            self.losses = {'train':[], 'test':[]}
        else:
            self.losses = {'train':[]}

    def make_dataloader(self, X, y=None, shuffle=True, num_workers=8):
        '''
        Wraps a dataloader to iterate over (X, y)
        '''

        kwargs = {}
        if self.allow_cuda:
            kwargs = {'num_workers': 4, 'pin_memory': True}
        else:
            kwargs = {'num_workers': 4}

        if y is None:
            y = pd.Series([0] * X.shape[0])

        X, y = check_X_y(X, y.values.ravel())
        X = pd.DataFrame(X)
        y = pd.Series(y)

        loader = data_utils.DataLoader(
            data_utils.TensorDataset(
                torch.from_numpy(X.values).float(),
                torch.from_numpy(y.values).float()
            ),
            batch_size=self.mb_size,
            shuffle=shuffle,
            **kwargs)

        return loader

    def split_train_test(self):
        '''
        Splits Train-Test partitions
        '''

        if (self.train_size < 1) and (self.train_size > 0):
            X_train, X_test = train_test_split(
                self.X,
                train_size=self.train_size,
                test_size=1-self.train_size)
        else:
            X_train = self.X
            X_test = self.X

        self.X_train = X_train
        self.X_test = X_test

    def init_layers(self):
        # Encoder
        self.Encoder = torch.nn.Sequential(
            torch.nn.Linear(self.X_dim, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.z_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.p_dropout)
        )

        # Decoder
        self.Decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.X_dim),
            torch.nn.Sigmoid()
        )

        self.Encoder_solver = optim.Adam(
            self.Encoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

        self.Decoder_solver = optim.Adam(
            self.Decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

    def reset_grad(self):
        self.Encoder.zero_grad()
        self.Decoder.zero_grad()

    def fit(self, X):
        '''
        '''
        if not self.is_fitted():
            self.X = pd.DataFrame(X).copy()
            self.X_dim = self.X.shape[1]

            self.split_train_test()
            self.init_layers()

            if self.allow_cuda:
                self.Encoder.cuda()
                self.Decoder.cuda()

            self.epoch_cnt = 0
            self.recon_train_loss = []
            self.fitted = True
            self.iterate_n_epochs(epochs=self.epochs)

    def iterate_n_epochs(self, epochs):
        '''
        Makes N training iterations
        '''
        self.criterion = nn.MSELoss()
        epoch_cnt = 0

        while(epoch_cnt < epochs):
            self.Encoder.train()
            self.Decoder.train()

            dataloader = self.make_dataloader(self.X_train)
            for batch_idx, (x, _) in enumerate(dataloader):
                self.reset_grad()

                if self.allow_cuda:
                    x = x.cuda()
                x = Variable(x)

                """ Reconstruction phase """
                x_encoded = self.Encoder(x)
                x_decoded = self.Decoder(x_encoded)

                recon_loss = self.criterion(x_decoded, x)

                recon_loss.backward()
                self.Encoder_solver.step()
                self.Decoder_solver.step()

            epoch_cnt += 1
            self.epoch_cnt += 1
            self.eval_train()

        self.Encoder.eval()
        self.Decoder.eval()

    def eval_train(self):
        self.Encoder.eval()
        self.Decoder.eval()

        if self.train_size < 1.:
            labels = ['train', 'test']
            datasets = [self.X_train, self.X_test]
        else:
            labels = ['train']
            datasets = [self.X_train]
        msg = ''
        for data_it, label in enumerate(labels):
            data = datasets[data_it]

            dataloader = self.make_dataloader(data)
            recon_loss = []
            for batch_idx, (x, _) in enumerate(dataloader):
                if self.allow_cuda:
                        x = x.cuda()
                x = Variable(x)

                """ Reconstruction phase """
                x_encoded = self.Encoder(x)
                x_decoded = self.Decoder(x_encoded)

                local_recon_loss = self.criterion(x_decoded, x)
                recon_loss.append(local_recon_loss.item())

            losses = sum(recon_loss) / len(recon_loss)
            msg += '[%s] %s %s (%s)\t' % (
                self.epoch_cnt, label, losses, data.shape[0])

            self.losses[label].append(losses)

        if self.verbose:
            print(msg)

    def transform(self, X_sample):
        self.Encoder.eval()
        self.Decoder.eval()

        dataloader = self.make_dataloader(X_sample, shuffle=False)

        X_transformed = []
        for batch_idx, (X, _) in enumerate(dataloader):
            if self.allow_cuda:
                X = X.cuda()

            z_sample = self.Encoder(X)

            X_transformed.append(z_sample.detach().numpy())

        X_transformed = np.vstack(X_transformed)

        return X_transformed

    def inverse_transform(self, Z):
        self.Encoder.eval()
        self.Decoder.eval()

        # self = aae_model
        dataloader = self.make_dataloader(Z, shuffle=False)

        X_transformed = []
        for batch_idx, (z, _) in enumerate(dataloader):
            if self.allow_cuda:
                z = z.cuda()

            X_sample = self.Decoder(z)

            X_transformed.append(X_sample.detach().numpy())

        X_transformed = np.vstack(X_transformed)

        return X_transformed

    def is_fitted(self):
        return self.fitted
