#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:17:18 2020

@author: yiningma
"""

import os
import time
import argparse
import torch
# import numpy as np

def get_options(args=None):
    parser = argparse.ArgumentParser('FedPG-BF')

    ### overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', choices = ['HalfCheetah-v2', 'LunarLander-v2', 'CartPole-v1'], help='env name for the game')
    parser.add_argument('--eval_only', action='store_true', help='used only if to evaluate a model')
    parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use for starting')
    parser.add_argument('--render', action='store_true', help='render to view game')
    parser.add_argument('--mode', type=str, default='human')
    parser.add_argument('--multiple_run', type=int, default=1, help='number of repeat run')
    
    
    # training and validating
    parser.add_argument('--val_size', type=int, default=10, help='Number of episoid used for reporting validation performance')
    parser.add_argument('--val_max_steps', type=int, default=1000)
    
    
    # Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=10, help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=0, help = 'number of worker node that is Byzantine')
    parser.add_argument('--attack_type', type=str, default='filtering-attack', choices = ['random-action', 'reward-flipping', 'variance-attack', 'random-noise', 'filtering-attack'], help = 'the attack type of a Byzantine worker')
    
    
    # policy net
    parser.add_argument('--activation', default='Tanh')
    parser.add_argument('--output_activation', default='Tanh')
    
    
    # SVRG and SCSG
    parser.add_argument('--svrg', action='store_true', help='run SVRG')
    parser.add_argument('--scsg', action='store_true', help='run SCSG')

    
    ### Byzantine Filtering
    parser.add_argument('--with_filter', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=0.6)
    parser.add_argument('--sigma', type=float, default=0.06)


    # load models
    parser.add_argument('--load_path', default = None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--log_dir', default = 'logs')
    
    
    ### run_name for outputs
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')

    ### end of parameters
    opts = parser.parse_args(args)

    opts.use_cuda = False
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_saving else None
    opts.log_dir = os.path.join(
        f'{opts.log_dir}',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_tb else None
    
    if opts.env_name == 'CartPole-v1':
        opts.max_epi_len = 500  
        opts.max_trajectories = 5000
        opts.lr_model = 1e-3
        opts.do_sample_for_training = True
        opts.hidden_units = '16,16'
        opts.B = 16
        opts.b = 4
        opts.N = 3
        opts.Bmin = 12
        opts.Bmax = 20
        opts.gamma  = 0.999
        opts.min_reward = 0
        opts.max_reward = 600
        opts.alpha = 0.4
        opts.delta = 0.6
        opts.sigma = 0.06
        opts.activation = 'ReLU'
  
    elif opts.env_name == 'HalfCheetah-v2':
        opts.max_epi_len = 500  
        opts.max_trajectories = 1e4
        opts.lr_model = 8e-5 # 4e-3
        opts.hidden_units = '64,64'
        opts.do_sample_for_training = True
        opts.B = 48
        opts.b = 16
        opts.N = 3
        opts.Bmin = 46
        opts.Bmax = 50
        opts.gamma  = 0.995
        opts.min_reward = 0
        opts.max_reward = 4000
        opts.alpha = 0.4
        opts.delta = 0.6
        opts.sigma = 0.9
        opts.activation = 'Tanh'

# current best
#         opts.use_critic = False
#         opts.max_epi_len = 500  
#         opts.max_trajectories = 1e4
#         opts.lr_model = 1e-4 # 4e-3
#         opts.hidden_units = '64,64'
#         opts.do_sample_for_training = True
#         opts.B = 32
#         opts.b = 16
#         opts.N = 3
#         opts.Bmin = 26
#         opts.Bmax = 38
#         opts.gamma  = 0.995
#         opts.min_reward = -1000
#         opts.max_reward = 4000
#         opts.alpha = 0.4
#         opts.delta = 0.6
#         opts.sigma = 0.9
#         opts.activation = 'Tanh'
#         opts.epsilon = 0.0

    if opts.env_name == 'LunarLander-v2':
        opts.max_epi_len = 1000  
        opts.max_trajectories = 1e4
        opts.lr_model = 1e-3 # 8e-4
        opts.do_sample_for_training = True
        opts.hidden_units = '64,64'
        opts.B = 32
        opts.b = 8 # 24
        opts.N = 3
        opts.Bmin = 26
        opts.Bmax = 38
        opts.gamma  = 0.99
        opts.min_reward = -1000
        opts.max_reward = 300  
        opts.sigma = 0.07
        opts.activation = 'Tanh'

## current running
#         opts.use_critic = False
#         opts.max_epi_len = 1000  
#         opts.max_trajectories = 1e4
#         opts.lr_model = 1e-3 # 8e-4
#         opts.do_sample_for_training = True
#         opts.hidden_units = '64,64'
#         opts.B = 32
#         opts.b = 16 # 24
#         opts.N = 2
#         opts.Bmin = 28
#         opts.Bmax = 34
#         opts.gamma  = 0.99
#         opts.min_reward = -1000
#         opts.max_reward = 300  
#         opts.epsilon = 0.0 # 0.2, 0.03
#         opts.sigma = 0.07
#         opts.activation = 'Tanh'

## best so far
#         opts.use_critic = False
#         opts.max_epi_len = 500  
#         opts.max_trajectories = 100000
#         opts.lr_model = 5e-4
#         opts.do_sample_for_training = True
#         opts.hidden_units = '64,64' #'16,16'
#         opts.B = 32
#         opts.b = 24
#         opts.N = 2
#         opts.Bmin = 30
#         opts.Bmax = 34
#         opts.gamma  = 0.99
#         opts.min_reward = -2000
#         opts.max_reward = 300  
#         opts.epsilon = 0.0
#         opts.activation = 'Tanh'

    # if opts.with_filter:
    #     assert opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))) <= 2 * opts.num_worker / opts.delta, \
    #         print( opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))), 2 * opts.num_worker / opts.delta)
    #     assert 2 * opts.num_worker / opts.delta <= np.exp(opts.B/2), \
    #         print(2 * opts.num_worker / opts.delta, np.exp(opts.B/2))

    assert opts.svrg + opts.scsg <= 1
    print('run vpg\n' if opts.svrg + opts.scsg == 0 else ('run scsg\n' if opts.scsg else 'run svrg\n'))
    
    assert not ( (not opts.svrg ) and (not opts.scsg) and opts.with_filter), 'donot support this currently'
    return opts
