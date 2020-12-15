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
    parser = argparse.ArgumentParser(
                                        # Byzantine Distributed RL
                                    )

    ### overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='Hopper-v2',# choices = ['Hopper-v2', 'Swimmer-v2', 'CartPole-v1'], 
                        help='env name for the game')
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--do_saving', action='store_false', help='Enable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard logging')
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--render', action='store_true', help='render to view game')
    
    
    # training and validating
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of episoid used for reporting validation performance')
    
    
    # Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=10, help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=0, help = 'number of worker node that is Byzantine')
    parser.add_argument('--attack_type', type=str, default='', choices = [''], help = 'the attack type of a Byzantine worker')
    
    
    # policy net
    parser.add_argument('--hidden_units', default = '16')
    parser.add_argument('--activation', default='Tanh')
    parser.add_argument('--output_activation', default='Identity')
    
    
    # SVRG and SCSG
    parser.add_argument('--svrg', action='store_true', help='run SVRG')
    parser.add_argument('--scsg', action='store_false', help='run SCSG')
    
    parser.add_argument('--Bmin', type=int, default=12,
                        help='Number of min batch per epoch for worker node during training (SCSG)')
    parser.add_argument('--Bmax', type=int, default=20,
                        help='Number of max batch per epoch for worker node during training (SCSG)')
    parser.add_argument('--B', type=int, default=16,
                        help='Number of batch per epoch for worker node during training (SVRG)')
    parser.add_argument('--N', type=int, default=4,
                        help='Number of batch per epoch for master node during training (SVRG)')
    parser.add_argument('--b', type=int, default=4,
                        help='Number of batch per epoch for master node during training')
    
    # REINFORCE
    parser.add_argument('--gamma', type=float, default=0.99)

    ### Byzantine Filtering
    parser.add_argument('--with_filter', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=0.6)
    parser.add_argument('--sigma_square', type=float, default=0.6)


    # resume and load models
    parser.add_argument('--load_path', default = None,
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, #'C:/Users/e0408674/Desktop/Byzantine-RL/codes/outputs/CartPole-v1/worker10_byzantine0_/run_name_20201215T233536/epoch-1.pt',
                        help='Resume from previous checkpoint file')

    
    ### run_name for outputs
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')


    ### end of parameters
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if opts.do_saving else None
    opts.log_dir = os.path.join(
        'logs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_tb else None
            
    if opts.env_name == 'CartPole-v1':
        opts.max_epi_len = 500   
        opts.max_trajectories = 3000
        opts.lr_model = 1e-3
        opts.hidden_units = '16,16'
    else:
        opts.max_epi_len = 1000   
        opts.max_trajectories = 10000
        opts.lr_model = 1e-4
        opts.hidden_units = '64,64,64,'
        opts.gamma = 0.995
    
    # if opts.with_filter:
    #     assert opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))) <= 2 * opts.num_worker / opts.delta, \
    #         print( opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))), 2 * opts.num_worker / opts.delta)
    #     assert 2 * opts.num_worker / opts.delta <= np.exp(opts.B/2), \
    #         print(2 * opts.num_worker / opts.delta, np.exp(opts.B/2))
    
    assert opts.svrg + opts.scsg <= 1
    print('run vpg\n' if opts.svrg + opts.scsg == 0 else ('run scsg\n' if opts.scsg else 'run svrg\n'))
    
    assert not ( (not opts.svrg ) and (not opts.scsg) and opts.with_filter), 'donot support this currently'
    return opts