import os
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from worker import Worker
from utils import torch_load_cpu, get_inner_model
from sklearn import metrics

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    dist = dist.sqrt()
    return dist


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
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation
        ).to(opts.device)
        
        # figure out a copy of the master node for importance sampling purpose
        self.old_master = Worker(
                id = -1,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation
        ).to(opts.device)
        
        # figure out all the actors
        self.workers = []
        self.true_Byzantine = []
        for i in range(self.world_size):
            self.true_Byzantine.append(True if i < opts.num_Byzantine else False)
            self.workers.append( Worker(
                                    id = i+1,
                                    is_Byzantine = True if i < opts.num_Byzantine else False,
                                    env_name = opts.env_name,
                                    gamma = opts.gamma,
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
        
        # parameters of running
        opts = self.opts

        # for storing number of trajectories sampled
        step = 0
        
        # Start the training loop
        for epoch in range(opts.epoch_start, opts.epoch_end):
            # Turn model into training mode
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            self.train()
            
            # setup lr_scheduler
            self.lr_scheduler.step(epoch)
            print("Training with lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)
            
            # some emplty list for training and logging purpose
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []
            
            # distribute current x to all workers and collect the gradient from workers
            param = get_inner_model(self.master.logits_net).state_dict()
            for worker in tqdm(self.workers, desc='Worker node'):

                # distribute current parameters
                worker.load_param_from_master(param)
                
                # get returned gradients and info from all agents
                grad, loss, rets, lens = worker.train_one_epoch(opts.B, opts.device)
                
                # store all values
                gradient.append(grad)
                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)
       
            # make the old policy as a copy of the current master node
            self.old_master.load_param_from_master(param)
            
            # do filter step to detect Byzantine worker on master node if needed
            if opts.with_filter:
                
                # calculate C, Variance Bound V, thresold, and alpha
                C = 2 * np.log(2 * opts.num_worker / opts.delta)
                V = opts.V
                threshold = 2 * V * np.sqrt(C / opts.B)
                alpha = opts.alpha
            
                # flatten the gradient vectors of each worker and put them together, shape [num_worker, -1]
                mu_vec = None
                for idx,item in enumerate(self.old_master.parameters()):
                    # stack gradient[idx] from all worker nodes
                    grad_item = []
                    for i in range(self.world_size):
                         grad_item.append(gradient[i][idx])
                    grad_item = torch.stack(grad_item).view(self.world_size,-1)
                
                    # concat stacked grad vector
                    if mu_vec is None:
                        mu_vec = grad_item.clone()
                    else:
                        mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)
                    
                # calculate the norm distance between each worker's gradient vector, shape [num_worker, num_worker]
                dist = euclidean_dist(mu_vec, mu_vec)
                
                # find the index of "compact" worker's gradient such that |dist <= threshold| > 0.5 * num_worker
                mu_med_vec = None
                k_prime = (dist <= threshold).sum(-1) > (0.5 * self.world_size)
                
                # computes the vector median of the gradients, mu_med_vec, and
                # filter the gradients it believes to be Byzantine and store the index of non-Byzantine graidents in Good_set
                if torch.sum(k_prime) > 0:
                    mu_med_vec = torch.median(mu_vec[k_prime],0)[0].view(1,-1)
                    Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 2 * threshold
                else:
                    Good_set = k_prime # if median vector can not be calculated, skip this step, k_prime is emplty (i.e., all False)
                
                # avoid the scenarios that Good_set is empty or can have |Gt| < (1 − α)K.
                if torch.sum(Good_set) < (1 - alpha) * self.world_size or torch.sum(Good_set) == 0:
                    # re-calculate vector median of the gradients
                    k_prime = (dist < 2 * V).sum(-1) > (0.5 * self.world_size)
                    mu_med_vec = torch.median(mu_vec[k_prime],0)[0].view(1,-1)
                    # re-filter with new vector median
                    Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 4 * V
            
            # else will treat all nodes as non-Byzantine nodes
            else:
                Good_set = torch.ones(self.world_size,1).to(opts.device).bool()
            
            # calculate number of good gradients for logging
            N_good = torch.sum(Good_set)
            
            # aggregate all detected non-Byzantine gradients to get mu
            if N_good > 0:
                mu = []
                for idx,item in enumerate(self.old_master.parameters()):
                    grad_item = []
                    for i in range(self.world_size):
                        if Good_set[i]: # only aggregate non-Byzantine gradients
                            grad_item.append(gradient[i][idx])
                    mu.append(torch.stack(grad_item).mean(0))
            else: # if still all nodes are detected to be Byzantine, set mu to be None
                mu = None
            
            # perform gradient update in master node
            grad_array = [] # store gradients for logging
            # with svrg or graident descent
            if not opts.no_svrg:
                # for n=1 to Nt ~ Geom(B/B+b) do grad update
                B = opts.B
                b = opts.b
                N_t = np.random.geometric(p= 1 - B/(B + b))
                for n in tqdm(range(N_t), desc = 'Master node'):
                   
                    # calculate new gradient in master node
                    self.optimizer.zero_grad()
                
                    # sample one trajectory using the old policy (\theta_0) of master node
                    weights, old_logp, batch_rets, batch_lens, batch_states, batch_actions = self.old_master.collect_experience_for_training(b, 
                                                                                                                                 opts.device, 
                                                                                                                                 record = True)
                
                    # calculate gradient for the old policy (\theta_0)
                    loss_old = -(old_logp * weights).mean()
                    self.old_master.logits_net.zero_grad()
                    loss_old.backward()
                    grad_old = [item.grad for item in self.old_master.parameters()]
                    
                    # get the new log_p with the latest policy (\theta_n) but fixing the actions to be the same as the sampled trajectory
                    new_logp = []
                    for idx, obs in enumerate(batch_states):
                        # act in the environment with the fixed action
                        _, new_log_prob = self.master.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(opts.device), 
                                                                     fixed_action = batch_actions[idx])
                        # store in the old_logp
                        new_logp.append(new_log_prob)
                    new_logp = torch.stack(new_logp)
                    
                    # calculate gradient for the latest policy (\theta_n)
                    loss = -(new_logp * weights).mean()
                    self.master.logits_net.zero_grad()
                    loss.backward()
                    
                    # Finding the ratio (pi_theta / pi_theta__old):
                    ratios = torch.exp(new_logp.detach().sum() - old_logp.detach().sum())
                    
                    # adjust and set the gradient for latest policy (\theta_n)
                    for idx,item in enumerate(self.master.parameters()):
                        item.grad = item.grad - ratios * grad_old[idx] + mu[idx] if mu else item.grad # if mu is None, use grad from master 
                        grad_array += (item.grad.data.view(-1).cpu().tolist())
                
                    # take a gradient step
                    self.optimizer.step()
            else:
                # perform gradient descent with mu vector
                for idx,item in enumerate(self.master.parameters()):
                    item.grad = item.grad + mu[idx]
                    grad_array += (item.grad.data.view(-1).cpu().tolist())
            
                # take a gradient step
                self.optimizer.step()    
            
            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens)))
            
            # Logging to tensorboard
            if(tb_logger is not None):
                # current step: number of trajectories sampled
                step += (B + b * N_t)
                
                # training log
                tb_logger.add_scalar('train/total_rewards', np.mean(batch_rets), step)
                tb_logger.add_scalar('train/epi_length', np.mean(batch_lens), step)
                tb_logger.add_scalar('train/loss', np.mean(batch_loss), step)
                # grad log
                tb_logger.add_scalar('grad/grad', np.mean(grad_array), step)
                # optimizer log
                tb_logger.add_scalar('params/lr', self.optimizer.param_groups[0]['lr'], step)
                tb_logger.add_scalar('params/N_t', N_t, step)
                # Byzantine filtering log
                if opts.with_filter:
                    
                    y_true = self.true_Byzantine
                    y_pred = (~ Good_set).view(-1).cpu().tolist()
                    
                    dist_Byzantine = dist[:opts.num_Byzantine][:,:opts.num_Byzantine]
                    dist_good = dist[opts.num_Byzantine:][:,opts.num_Byzantine:]
    
                    tb_logger.add_scalar('grad/grad_norm_Byzantine_mean', torch.mean(dist_Byzantine), step)
                    tb_logger.add_scalar('grad/grad_norm_Byzantine_max', torch.max(dist_Byzantine), step)
                    tb_logger.add_scalar('grad/grad_norm_Good_mean', torch.mean(dist_good), step)
                    tb_logger.add_scalar('grad/grad_norm_Good_max', torch.max(dist_good), step)
        
                    tb_logger.add_scalar('Byzantine/N_good_pred', N_good, step)
                    tb_logger.add_scalar('Byzantine/threshold', threshold, step)
                    
                    tb_logger.add_scalar('Byzantine/precision', metrics.precision_score(y_true, y_pred), step)
                    tb_logger.add_scalar('Byzantine/recall', metrics.recall_score(y_true, y_pred), step)
                    tb_logger.add_scalar('Byzantine/f1_score', metrics.f1_score(y_true, y_pred), step)
                
                
                # do validating
                self.start_validating(tb_logger, step)
            
            # save current model
            if opts.do_saving:
                self.save(epoch)
                
    
    # validate the new model   
    def start_validating(self,tb_logger = None, id = 0, render = False):
        print('\nValidating...', flush=True)
        
        val_ret = 0.0
        val_len = 0.0
        
        for _ in range(self.opts.val_size):
            epi_ret, epi_len = self.master.rollout(self.opts.device, max_epi = self.opts.max_epi_len, render = render)
            val_ret += epi_ret
            val_len += epi_len
        
        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size
        
        print('\nGradient step: %3d \t return: %.3f \t ep_len: %.3f'%
                (id,  np.mean(val_ret), np.mean(val_len)))
        
        if(tb_logger is not None):
            tb_logger.add_scalar('validate/total_rewards', np.mean(val_ret), id)
            tb_logger.add_scalar('validate/epi_length', np.mean(val_len), id)
            tb_logger.close()