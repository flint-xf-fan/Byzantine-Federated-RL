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
    parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--RL_agent', default='DPG', choices = ['DPG'])
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
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=5,
                        help="Batch size to use during evaluation")
    
    
    ### logs to tensorboard and screen
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')

    ### outputs
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.RL_agent, opts.env_name),
        opts.run_name
    ) if not opts.no_saving else None
    return opts