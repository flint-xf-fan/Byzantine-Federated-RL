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
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', choices = ['MountainCar-v0', 'CartPole-v1'], 
                        help='env name for the game')
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--do_saving', action='store_true', help='Enable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard logging')
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--render', action='store_true', help='render to view game')
    
    # Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=5, help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=2, help = 'number of worker node that is Byzantine')
    
    # logits_net net (mlp currently)
    parser.add_argument('--hidden_units', default = '8,')
    parser.add_argument('--activation', default='Tanh')
    parser.add_argument('--output_activation', default='Identity')

    # resume and load models
    parser.add_argument('--load_path', default = None,
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None,
                        help='Resume from previous checkpoint file')

    ### training
    parser.add_argument('--B', type=int, default=16,
                        help='Number of batch per epoch for worker node during training')
    parser.add_argument('--b', type=int, default=8,
                        help='Number of batch per epoch for master node during training')
    parser.add_argument('--max_epi_len', type=int, default=500)
     
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Strat at epoch #')
    parser.add_argument('--epoch_end', type=int, default=100,
                        help='End at epoch #')
    parser.add_argument('--lr_model', type=float, default=1e-2, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay per epoch')
    parser.add_argument('--no_svrg', action='store_true', help='Disable SVRG')
    parser.add_argument('--gamma', type=float, default=0.99)



    ### Byzantine Filtering
    parser.add_argument('--with_filter', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.4)
    
    parser.add_argument('--delta', type=float, default=0.6)
    parser.add_argument('--V', type=float, default=0.025)



    ### validation   
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of episoid used for reporting validation performance')
    
    ### run_name for outputs
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')


    ### end of parameters
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}".format(opts.num_worker, opts.num_Byzantine),
        opts.run_name
    ) if opts.do_saving else None
    opts.log_dir = os.path.join(
        # 'logs',
        'logs_test',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}".format(opts.num_worker, opts.num_Byzantine),
        opts.run_name
    ) if not opts.no_tb else None
            
    # if opts.with_filter:
    #     assert opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))) <= 2 * opts.num_worker / opts.delta, \
    #         print( opts.delta * opts.B / (np.exp(2 * (1 - 2 * opts.delta))), 2 * opts.num_worker / opts.delta)
    #     assert 2 * opts.num_worker / opts.delta <= np.exp(opts.B/2), \
    #         print(2 * opts.num_worker / opts.delta, np.exp(opts.B/2))
    
    assert not (opts.no_svrg and opts.with_filter), 'donot support this currently'
    return opts