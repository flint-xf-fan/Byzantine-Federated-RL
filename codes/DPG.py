import os
from tqdm import tqdm

import torch
import numpy as np
import torch.optim as optim

from actor_network import Actor
from utils import torch_load_cpu, get_inner_model

class DPG:
    def __init__(self, opts):
        
        # figure out the options
        self.opts = opts
        
        # figure out the master
        self.master = Actor(
                0,
                opts.env_name,
                opts.hidden_units, 
                opts.activation, 
                opts.output_activation
        ).to(opts.device)
        
        # figure out all the actors
        self.workers = []
        self.num_worker = opts.num_worker
        for i in range(opts.num_worker):
            self.workers.append(Actor(
                    i + 1,
                    opts.env_name,
                    opts.hidden_units, 
                    opts.activation, 
                    opts.output_activation
            ).to(opts.device))
        print(f'{opts.num_worker} workers initilized.')
        
        if not opts.eval_only:
            # figure out the optimizer
            self.optimizer = optim.Adam(
                [{'params': self.master.parameters(), 'lr': opts.lr_model}])
            # learning rate decay
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.lr_decay ** (epoch))
        
        # parallel actor
#        if opts.use_cuda and torch.cuda.device_count() > 1:
#            self.actor = torch.nn.DataParallel(self.actor)
    
    def load(self, load_path):
        
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.master)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})
        
        if not self.opts.eval_only:
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))
        
    
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'master': get_inner_model(self.master).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    
    def eval(self):
        # turn model to eval mode
        self.master.eval()
        
    def train(self):
        # turn model to trainig mode
        self.master.train()
    
    def start_training(self, tb_logger = None):
        
        opts = self.opts

        # Start the actual training loop
        for epoch in range(opts.epoch_start, opts.epoch_end):
            # Training mode
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            self.train()
            
            # lr_scheduler
            self.lr_scheduler.step(epoch)
            print("Training with lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)
            
            # training starts here
            param = get_inner_model(self.master).state_dict()
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []
            
            for agent in tqdm(self.workers,desc='Worker node'):
                
                # distribute current parameters
                agent.load_net_param(param)
                
                # get returned gradients and info from all agents
                grad, loss, rets, lens = agent.train_one_epoch(opts.batch_size, opts.device)
                gradient.append(grad)
                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)
                
            
            # calculate new gradient in master node
            self.optimizer.zero_grad()
            
            for idx,item in enumerate(self.master.parameters()):
                grad_item = []
                for i in range(self.num_worker):
                     grad_item.append(gradient[i][idx])
                item.grad = torch.stack(grad_item).mean(0)
        
            self.optimizer.step()
            
            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens)))
            
            # Logging to tensorboard
#            if(not opts.no_tb):
#              pass
            
            # save current model
#            self.save(epoch)
                
    
    # validate the new model   
    def start_validating(self,tb_logger = None):
        pass
