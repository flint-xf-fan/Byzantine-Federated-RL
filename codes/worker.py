import torch
import numpy as np
import gym
from gym.spaces import Discrete, Box
from policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_inner_model
from copy import deepcopy
import math
import highway_env

class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = None,
                 max_epi_len = 0,
                 opts = None
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        
        assert opts is not None
        
        if opts.highway:

            if opts.discrete:
                
                self.env.configure({
                        "observation": {
                            "type": "Kinematics",
                            "vehicles_count": 8,
                            "absolute": False
                        }
                    })
                
                obs_dim = self.env.observation_space.shape[1] * 8
                n_acts = self.env.action_space.n
            
                hidden_sizes = list(eval(hidden_units))
                self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
                
                self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
            else:
                raise NotImplementedError()

        
        else:
        
            obs_dim = self.env.observation_space.shape[0]
            if isinstance(self.env.action_space, Discrete):
                n_acts = self.env.action_space.n
            else:
                n_acts = self.env.action_space.shape[0]
            
            hidden_sizes = list(eval(hidden_units))
            self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
            
            if isinstance(self.env.action_space, Discrete):
                self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
            else:
                if self.env_name == 'Humanoid-v2':
                    self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation, geer = 0.4)
                else:
                    self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})
    
    def rollout(self, device, render = False, env = None, obs = None, sample = True):
        
        if env is None and obs is None:
            env = gym.make(self.env_name)
            obs = env.reset()
        done = False  
        ep_rew = []
        while not done:
            if render:
                env.render()
            
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ = env.step(action)
            ep_rew.append(rew)

        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False, sample = True):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment            
            act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
            obs, rew, done, info = self.env.step(act)
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
                
                batch_weights += returns

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device, sample):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(B, device, sample = sample)
        
        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()
        
        # determine if the agent is byzantine
        if self.is_Byzantine and self.attack_type is not None:
            # return wrong gradient with noise
            grad = []
            for item in self.parameters():
                # rnd_11 = (torch.rand(item.grad.shape, device = item.device) * 2. - 1.)
                # rnd_o = ((torch.rand(item.grad.shape, device = item.device) > 0.5).float())
                # grad.append(item.grad + item.grad * rnd_11 * rnd_o * 2)  
                
                # item.grad[item.grad > item.grad.mean()] = -item.grad[item.grad > item.grad.mean()] * 2
                # grad.append(item.grad)  

                # rnd = torch.rand(item.grad.shape, device = item.device) * (item.grad.max().data - item.grad.min().data) + item.grad.min().data
                # rnd2 = ((torch.rand(item.grad.shape, device = item.device) > 0.5).float())
                # grad.append(item.grad + rnd * rnd2 * 5)    
                # grad.append(rnd * 2 * rnd2)    

                if self.attack_type == 'sign-flipping':
                    grad.append( - item.grad)  

                elif self.attack_type == 'zero-gradient':
                    grad.append( 0 * item.grad)

                elif self.attack_type == 'random-noise':
                    rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) 
                    grad.append( item.grad + rnd)  
                
                elif self.attack_type == 'detect-attack':
                    grad.append(item.grad)
                
                else: raise NotImplementedError()

    
        else:
            # return true gradient
            grad = [item.grad for item in self.parameters()]
        
        # report the results to the agent for training purpose
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)


    def to(self, device):
        self.logits_net.to(device)
        return self
    
    def eval(self):
        self.logits_net.eval()
        return self
        
    def train(self):
        self.logits_net.train()
        return self
    
    def parameters(self):
        return self.logits_net.parameters()