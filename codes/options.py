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

def get_options(args=None):
    parser = argparse.ArgumentParser(
                                        # Byzantine Distributed RL
                                    )

    ### overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', 
                        help='env name for the game')
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--do_saving', action='store_true', help='Enable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard logging')
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--render', action='store_true', help='render to view game')
    
    # important parameters
    parser.add_argument('--num_worker', type=int, default=10, help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=2, help = 'number of worker node that is Byzantine')
    
    # logits_net net (mlp currently)
    parser.add_argument('--hidden_units', default = '32,')
    parser.add_argument('--activation', default='Tanh')
    parser.add_argument('--output_activation', default='Identity')

    # resume and load models
    parser.add_argument('--load_path', default = None,
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None,
                        help='Resume from previous checkpoint file')

    ### training
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Number of batch per epoch during training')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Strat at epoch #')
    parser.add_argument('--epoch_end', type=int, default=50,
                        help='End at epoch #')
    parser.add_argument('--lr_model', type=float, default=1e-2, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay per epoch')


    ### validation   
    parser.add_argument('--val_size', type=int, default=5,
                        help='Number of episoid used for reporting validation performance')
    parser.add_argument('--max_epi_len', type=int, default=5000)
    
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
            
    return opts