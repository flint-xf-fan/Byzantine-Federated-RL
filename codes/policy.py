#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:26:40 2020

@author: yiningma
"""
import numpy as np
import torch
import math
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class CnnPolicy(nn.Module):
    def __init__(self,
                 sizes,
                 activation = 'Tanh',
                 output_activation = 'Identity'):
        
        super(CnnPolicy, self).__init__()
        
        # store parameters
        self.activation = activation
        self.output_activation = output_activation
        
        if activation == 'Tanh':
            self.activation = nn.Tanh
        elif activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError
            
        if output_activation == 'Identity':
            self.output_activation = nn.Identity()
        elif output_activation == 'Tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'ReLU':
            self.output_activation = nn.ReLU()
        elif output_activation == 'Softmax':
            self.output_activation = nn.Softmax()
        else:
            raise NotImplementedError
            
        self.conv1 = nn.Conv2d(4, 8, 5)
        self.conv2 = nn.Conv2d(8, 12, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3888, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 16)
        self.fc3 = nn.Linear(16, sizes[-1], bias = False)
            
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)   
            
    def forward(self, obs, sample = True, fixed_action = None):
        """
        :param input: (obs) input observation
        :return: action
        """
        
#         assert obs.shape == (84,84,4)
        x = obs
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = self.output_activation(x) * 10.
        # print(x)
        
        # get the policy dist
        policy = Categorical(logits=x)
        # print(policy.probs)
        
        # take the pre-set action
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        
        elif sample:
            try:
                action = policy.sample()
            except:
                print(x,obs)
        # take greedy action
        else:
            action = policy.probs.argmax()
        
        return action.item(), policy.log_prob(action)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MlpPolicy(nn.Module):
    def __init__(self,
                 sizes,
                 activation = 'Tanh',
                 output_activation = 'Identity'):
        
        super(MlpPolicy, self).__init__()
        
        # store parameters
        self.activation = activation
        self.output_activation = output_activation
        
        if activation == 'Tanh':
            self.activation = nn.Tanh
        elif activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError
            
        if output_activation == 'Identity':
            self.output_activation = nn.Identity
        elif output_activation == 'Tanh':
            self.output_activation = nn.Tanh
        elif output_activation == 'ReLU':
            self.output_activation = nn.ReLU
        elif output_activation == 'Softmax':
            self.output_activation = nn.Softmax
        else:
            raise NotImplementedError
            
        # make policy network
        self.sizes = sizes
        self.logits_net = mlp(self.sizes, self.activation, self.output_activation)
    
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, obs, sample = True, fixed_action = None):
        """
        :param input: (obs) input observation
        :return: action
        """
        
        obs = obs.view(-1)
        
        # forward pass the policy net
        logits = self.logits_net(obs)
        
        # get the policy dist
        policy = Categorical(logits=logits)
        
        # take the pre-set action
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        
        elif sample:
            try:
                action = policy.sample()
            except:
                print(logits,obs)
        # take greedy action
        else:
            action = policy.probs.argmax()
        
        return action.item(), policy.log_prob(action)
        

class DiagonalGaussianMlpPolicy(nn.Module):
    def __init__(self,
                 sizes,
                 activation = 'Tanh',
                 output_activation = 'Tanh',
                 geer = 1):
        
        super(DiagonalGaussianMlpPolicy, self).__init__()
        
        # store parameters
        self.activation = activation
        self.output_activation = output_activation
        
        if activation == 'Tanh':
            self.activation = nn.Tanh
        elif activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError
            
            
        # make policy network
        self.sizes = sizes
        self.logits_net = mlp(self.sizes[:-1], self.activation)
        self.mu_net = nn.Linear(self.sizes[-2],self.sizes[-1], bias = False)
        self.log_sigma_net = nn.Linear(self.sizes[-2],self.sizes[-1], bias = False)
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        self.init_parameters()
        
        self.geer = geer

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, obs, sample = True, fixed_action = None):
        """
        :param input: (obs) input observation
        :return: action
        """
        
        # forward pass the policy net
        h_mu = self.logits_net(obs) 

        # get the mu
        mu = torch.tanh(self.mu_net(h_mu)) * self.geer
        
        # get the sigma
        log_sigma = torch.clamp(self.log_sigma_net(h_mu), self.LOG_STD_MIN, self.LOG_STD_MAX)
        sigma = torch.tanh(log_sigma.exp())
        
        # get the policy dist
        policy = Normal(mu, sigma)
        
        # take the pre-set action
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        else:
            if sample:
                action = policy.sample()
                    
            else:
                action = mu.detach()
        
        return action.numpy(), policy.log_prob(action).sum()
    
    
class LinearCritic(nn.Module):
    def __init__(self,
                 sizes,):
        
        super(LinearCritic, self).__init__()
        
        self.regr = linear_model.LinearRegression()
        self.regr.intercept_ = 0
        self.regr.coef_ = np.zeros(sizes[0] * 2 + 4) + 1/sizes[0] * 2 + 4
        

    def fit(self, X, y):
        
        self.regr.fit(X, y)
        y_pred =  self.regr.predict(X)
        
        mse_loss = mean_squared_error(y, y_pred)
        
        return mse_loss
        
    def predict(self, obs, t):
        obs = obs.view(-1)
        obs = torch.cat((obs,
                         obs * obs,
                         torch.tensor([t * 0.1]).float(),
                         torch.tensor([t * 0.1]).pow(2).float(),
                         torch.tensor([t * 0.1]).pow(3).float(),
                         torch.tensor([1.]).float())).view(1,-1).numpy()
        
        return obs, self.regr.predict(obs)[0]
    
    def get_parameters(self):
        return {'intercept_': self.regr.intercept_, 'coef_': self.regr.coef_}
        
        
    def set_parameters(self, param_dict):
        self.regr.intercept_ = param_dict['intercept_']
        self.regr.coef_ = param_dict['coef_']
        