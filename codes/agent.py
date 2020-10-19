import os
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from worker import Worker
from utils import torch_load_cpu, get_inner_model

class Agent:
    
    def __init__(self, opts):
        
        # figure out the options
        self.opts = opts
        
        # setup arrays for distrubuted RL
        self.world_size = opts.num_worker
        
        # figure out the master
        self.master = Worker(
                id = 0,
                is_Byzantine = False,
                env_name = opts.env_name,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation
        ).to(opts.device)
        
        # figure out all the actors
        self.workers = []
        for i in range(self.world_size):
            self.workers.append( Worker(
                                    id = 0,
                                    is_Byzantine = False,
                                    env_name = opts.env_name,
                                    hidden_units = opts.hidden_units, 
                                    activation = opts.activation, 
                                    output_activation = opts.output_activation
                            ).to(opts.device))
        print(f'{opts.num_worker} workers initilized with {opts.num_Byzantine if opts.num_Byzantine >0 else "None"} of them are Byzantine.')
        
        if not opts.eval_only:
            # figure out the optimizer
            self.optimizer = optim.Adam(self.master.logits_net.parameters(), lr = opts.lr_model)
            # learning rate decay
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.lr_decay ** (epoch))
        
    
    def load(self, load_path):
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.master.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('master', {})})
        
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
                'master': get_inner_model(self.master.logits_net).state_dict(),
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
            param = get_inner_model(self.master.logits_net).state_dict()
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []
            
            for worker in tqdm(self.workers, desc='Worker node'):
                
                # distribute current parameters
                worker.load_param_from_master(param)
                
                # get returned gradients and info from all agents
                grad, loss, rets, lens = worker.train_one_epoch(opts.batch_size, opts.device)
                gradient.append(grad)
                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)
                
            
            # calculate new gradient in master node
            self.optimizer.zero_grad()
            
            for idx,item in enumerate(self.master.parameters()):
                grad_item = []
                for i in range(self.world_size):
                     grad_item.append(gradient[i][idx])
                item.grad = torch.stack(grad_item).mean(0)
        
            self.optimizer.step()
            
            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens)))
            
            # Logging to tensorboard
            if(tb_logger is not None):
                tb_logger.add_scalar('train/batch_loss', np.mean(batch_loss), epoch)
                tb_logger.add_scalar('train/total_rewards', np.mean(batch_rets), epoch)
                tb_logger.add_scalar('train/epi_length', np.mean(batch_lens), epoch)
                tb_logger.close()
            
            # do validating
            self.start_validating(tb_logger, epoch)
            
            # save current model
            if opts.do_saving:
                self.save(epoch)
                
    
    # validate the new model   
    def start_validating(self,tb_logger = None, id = 0):
        
        val_ret = 0.0
        val_len = 0.0
        
        for _ in range(self.opts.val_size):
            epi_ret, epi_len = self.master.rollout(self.opts.device, max_epi = self.opts.max_epi_len, render = False)
            val_ret += epi_ret
            val_len += epi_len
        
        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size
        
        if(tb_logger is not None):
            tb_logger.add_scalar('validate/total_rewards', np.mean(val_ret), id)
            tb_logger.add_scalar('validate/epi_length', np.mean(val_len), id)
            tb_logger.close()