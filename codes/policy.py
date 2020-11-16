#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:26:40 2020

@author: yiningma
"""
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Policy(nn.Module):
    def __init__(self,
                 sizes,
                 activation = 'Tanh',
                 output_activation = 'Identity'):
        
        super(Policy, self).__init__()
        
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
        
        