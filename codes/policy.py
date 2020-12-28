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
        else:
            raise NotImplementedError
            
        if output_activation == 'Identity':
            self.output_activation = nn.Identity
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
        
        # forward pass the policy net
        logits = self.logits_net(obs)
        
        # get the policy dist
        policy = Categorical(logits=logits)
        
        # take the pre-set action
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        
        elif sample:
            action = policy.sample()
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
        else:
            raise NotImplementedError
            
        if output_activation == 'Tanh':
            self.output_activation = nn.Tanh
        else:
            raise NotImplementedError
            
        # make policy network
        self.sizes = sizes
        self.logits_net = mlp(self.sizes, self.activation, self.output_activation)
        self.geer = geer
        
        self.sigma = nn.Parameter(torch.zeros(sizes[-1]))

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
        logits = self.logits_net(obs) * self.geer
        
        # get the mu
        mu = logits
        
        # get the sigma
        sigma = self.sigma.exp()
        
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
        
        return action, policy.log_prob(action).sum()