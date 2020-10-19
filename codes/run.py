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
from options import get_options
from torch.utils.tensorboard import SummaryWriter

def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))
    
    # setup tensorboard
    if not opts.no_tb:
        tb_writer = SummaryWriter(opts.log_dir)
    else:
        tb_writer = None

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    if opts.do_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if opts.do_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out the RL
    agent = Agent(opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)
    
    # Do validation only
    if opts.eval_only:
        agent.validate(tb_writer)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Starttraining here
        agent.start_training(tb_writer)
            


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    run(get_options())
