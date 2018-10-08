#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:31:08 2018

@author: Raul Sanchez-Vazquez
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures

from common.eval_utils import classification_report

class FMClassifier(nn.Module, BaseEstimator, RegressorMixin):
    """
    """
    def __init__(
            self,
            k=10,
            batch_size=32,
            lr=1e-3,
            alpha=1e-8,
            epochs=10,
            allow_cuda=False,
            test=None,
            val=None,
            dropout=.0,
            verbose=False):

        super(FMClassifier, self).__init__()

        self.k = k
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.epochs = epochs
        self.allow_cuda = allow_cuda
        self.dropout = dropout
        self.verbose = verbose
        self.test = test
        self.val = val

        # Internal usage
        self.embeddings = None
        self.poly_features = {}
        self.criterion = None
        self.optimizer = None
        self.train_epoch_loss = []
        self.aucs = {'train': []}

        if not self.test is None:
            self.aucs['test'] = []

        if not self.val is None:
            self.aucs['val'] = []

        self.poly_transform = PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False)

    def make_dataloader(self, X, y=None, shuffle=True):
        if y is None:
            y = pd.Series([0] * X.shape[0])

        loader = data_utils.DataLoader(
            data_utils.TensorDataset(
                torch.from_numpy(X.values).float(),
                torch.from_numpy(y.values).float()
            ), batch_size=self.batch_size, shuffle=shuffle)

        return loader

    def load(self, filename):
        """
        Loads a saved model
        """

        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """
        Saves as pickle object
        """

        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def init_poli_names(self):
        self.linear_features = self.X.columns.tolist()
        self.fm_features = []

        for c_i, c in enumerate(self.linear_features):
            for cc in self.linear_features[c_i + 1:]:
                self.fm_features.append('%s__%s' % (c, cc))

        self.poly_transform.fit(self.X)

    def init_embeddings(self):
        self.add_module('V', nn.Embedding(len(self.fm_features), self.k))

    def init_layers(self):
        self.W = nn.Linear(
            in_features=len(self.linear_features),
            out_features=1)

    def X_format(self, X):
        X_new = pd.DataFrame(
            self.poly_transform.transform(X),
            columns = self.linear_features + self.fm_features)

        return X_new

    def get_V_dot(self):
        n = len(self.linear_features)
        V_dot = []
        for v_i_idx in range(n):
            for v_j_idx in range(v_i_idx +1, n):
                v_i = Variable(self.V(
                    torch.from_numpy(np.array([v_i_idx])).long()
                ).reshape(self.k))

                v_j = Variable(self.V(
                    torch.from_numpy(np.array([v_j_idx])).long()
                ).reshape(self.k))

                V_dot.append(v_i.dot(v_j).reshape(1, 1))

        V_dot = torch.cat(V_dot)

        return V_dot

    def fit(self, X, y):
        """

        """

        self.X = X.copy()
        self.y = y.copy()

        # compute feature names
        self.init_poli_names()
        # Initializes V vectors
        self.init_embeddings()
        # Initializes regression layer
        self.init_layers()

        #self.iterate_n_epochs(self.epochs)

    def iterate_n_epochs(self, epochs):
        """
        epochs = 1
        """
        # Adam optimizer
        self.optimizer = torch.optim.SGD(
        #self.optimizer = torch.optim.Adagrad(
        #self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.alpha)

        self.train()

        criterion = torch.nn.MSELoss()
        train_loader = self.make_dataloader(
            self.X_format(self.X), self.y,
            shuffle=True)

        for epoch_idx in range(epochs):
            train_loss = []
            for batch_idx, (x, target) in enumerate(train_loader):

                self.optimizer.zero_grad()

                if self.allow_cuda:
                    x , target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target).float()

                output = self.forward(x)

                loss = criterion(
                    output.reshape(1, -1)[0],
                    target.float())

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

                if batch_idx % 20 == 0:
                    print(batch_idx)

            self.train_epoch_loss.append(sum(train_loss) / len(train_loss))
            self.eval_train()

    def forward(self, x):
        x_i = x[:, :len(self.linear_features)]
        x_ij = x[:, len(self.linear_features):]

        W_ij = self.get_V_dot()
        output = self.W(x_i) +  x_ij.mm(W_ij)
        output = torch.sigmoid(output)

        return output

    def predict_raw(self, X):
        """
        X = self.X
        """
        self.eval()

        loader = self.make_dataloader(
            self.X_format(X), shuffle=False)

        y_pred_raw = []
        for batch_idx, (x, _) in enumerate(loader):
            if self.allow_cuda:
                x = x.cuda()
            x = Variable(x)

            output = self.forward(x)

            if self.allow_cuda:
                output = output.cpu()
            y_pred_raw += output.data.numpy().flatten().tolist()

        y_pred_raw = np.array(y_pred_raw)

        return y_pred_raw

    def predict_proba(self, X):
        y_pred_raw = self.predict_raw(X)
        y_score = np.vstack([1 - y_pred_raw, y_pred_raw]).T

        return y_score

    def predict(self, X):
        y_pred_raw = self.predict_raw(X)
        y_pred = (y_pred_raw > .5).astype(int)

        return y_pred

    def eval_train(self):


        y_train_score = self.predict_proba(self.X)
        y_train_pred = (y_train_score[:, 1] < .5).astype(int)

        report_train = classification_report(
            y_true=self.y,
            y_pred=y_train_pred,
            y_score=y_train_score)

        self.aucs['train'].append(report_train['AUC'].iloc[-1])
        if self.verbose:
            print(report_train, '')

        if not self.test is None:
            y_test_score =  self.predict_proba(self.test[0])
            y_test_pred = (y_test_score[:, 1] < .5).astype(int)

            report_test = classification_report(
                y_true=self.test[1],
                y_pred=y_test_pred,
                y_score=y_test_score)

            self.aucs['test'].append(report_test['AUC'].iloc[-1])
            if self.verbose:
                print(report_test, '')

        if not self.val is None:
            y_val_pred = self.predict(self.val[0])
            y_val_score = self.predict_proba(self.val[0])

            report_val = classification_report(
                y_true=self.val[1],
                y_pred=y_val_pred,
                y_score=y_val_score)
            self.aucs['val'].append(report_val['AUC'].iloc[-1])
            if self.verbose:
                print(report_val, '')
