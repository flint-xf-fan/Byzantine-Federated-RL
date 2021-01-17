#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:27:02 2020

@author: yiningma
"""

import os
import json
import torch
import pprint
import numpy as np
from agent import Agent
from worker import Worker
from options import get_options
from torch.utils.tensorboard import SummaryWriter
from utils import get_inner_model

def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))
    
    # setup tensorboard
    if not opts.no_tb:
        tb_writer = SummaryWriter(opts.log_dir)
    else:
        tb_writer = None

    # Optionally configure tensorboard
    if opts.do_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Configure for multiple runs    
    assert opts.multiple_run > 0
    opts.seeds = np.arange(opts.multiple_run).tolist()

    # Save arguments so exact configuration can always be found
    if opts.do_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out the RL
    agent = Agent(opts)
    
    # Do validation only
    if opts.eval_only:
        # Set the random seed
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        # Load data from load_path
        if opts.load_path is not None:
            agent.load(opts.load_path)
        
        agent.start_validating(tb_writer, 0, opts.val_max_steps, opts.render, mode = opts.mode)
        
    else:
        for run_id in opts.seeds:
            # Set the random seed
            torch.manual_seed(run_id)
            np.random.seed(run_id)
            
            nn_parms_worker = Worker(
                id = 0,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
            ).to(opts.device)
                    
            model_actor = get_inner_model(agent.master.logits_net)
            model_actor.load_state_dict({**model_actor.state_dict(), **get_inner_model(nn_parms_worker.logits_net).state_dict()})
        
            # Starttraining here
            agent.start_training(tb_writer, run_id)
            if tb_writer:
                agent.log_performance(tb_writer)
            


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    run(get_options())
