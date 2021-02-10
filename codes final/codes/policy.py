#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

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
    
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, obs, sample = True, fixed_action = None):
        """
        :param input: observation
        :return: action, log_p(action)
        """
        
        obs = obs.view(-1)
        
        # forward pass the policy net
        logits = self.logits_net(obs)
        
        # get the policy dist
        policy = Categorical(logits=logits)
        
        # take the pre-set action if given
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        
        # take random action
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
        self.geer = geer
        self.logits_net = mlp(self.sizes[:-1], self.activation, nn.Identity)
        self.mu_net = nn.Linear(self.sizes[-2], self.sizes[-1], bias = False)
        self.log_sigma_net = nn.Linear(self.sizes[-2], self.sizes[-1], bias = False)
        self.LOG_SIGMA_MIN = -20
        self.LOG_SIGMA_MAX = -2
        
        # init parameters
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

        # forward pass the policy net
        logits = self.logits_net(obs)

        # get the mu
        mu = torch.tanh(self.mu_net(logits)) * self.geer

        # get the sigma
        sigma = torch.tanh(torch.clamp(self.log_sigma_net(logits), self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp())

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
        
        # avoid NaN
        ll = policy.log_prob(action)
        ll[ll < -1e5] = -1e5
        
        
        return action.numpy(), ll.sum()