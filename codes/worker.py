import torch
import numpy as np
import gym
from gym.spaces import Discrete, Box
from policy import MlpPolicy, DiagonalGaussianMlpPolicy, LinearCritic, CnnPolicy
from utils import get_inner_model, save_frames_as_gif
from copy import deepcopy
import math
import torch.optim as optim
from utils import env_wrapper
import pprint

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
            self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        if self.id == 1:
            print(self.logits_net)
        
        self.rew_max = None

    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})

    def rollout(self, device, max_steps = 1000, render = False, env = None, obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = self.env
            obs = env.reset()
            
        done = False  
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
                
            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ = env.step(action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        #print('reward:', np.sum(ep_rew), 'ep_len', len(ep_rew))
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False, sample = True, attack_type = None):
        #self.config()
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

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-action':
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int): # discrete action space
                    act_rnd = 0
                else: # continuous
                    act_rnd = np.zeros(len(self.env.action_space.sample()), dtype=np.float32) ################ a bug here for discrete action space but it works well for continuous! #######################
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = act_rnd)
            else:
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
           
            obs, rew, done, info = self.env.step(act)
            
#             if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping': 
#                 rew = -100 * rew # I see why this cannot work as desired. ############ remember that our data points are trajectries not individual timesteps!!!! you shold consider to alter the ep returns
                                 # or have to scale it. otherwise. it won't attack cos over the traj, the effect might just be reduce the value of the return. if it runs out sign is not changed, then no effect at all.
                                 # different reward can induce the same policy. this is one issue in inverse RL. #################################
#             elif attack_type is not None and self.attack_type == 'nosing-reward': # no need for this attack as we already have random noise
#                 if self.rew_max is None:
#                     self.rew_max = rew
#                 else:
#                     self.rew_max = max(np.abs(rew), self.rew_max)
#                     rew = np.random.rand() * 2 * self.rew_max - self.rew_max
    
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping':
                    ep_ret, ep_len = -2.5 * sum(ep_rews), len(ep_rews)
                else:
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32)

                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = self.env.reset(), False, [], 1


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device, sample):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(B, device, sample = sample, attack_type = self.attack_type)
        
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
                if self.attack_type == 'reward-flipping':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                
                elif self.attack_type == 'nosing-reward':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                    
                
                elif self.attack_type == 'random-action':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                    
                elif self.attack_type == 'variance-attack':
                    grad.append(item.grad)
                    # refer to agent.py to see attack

                elif self.attack_type == 'random-noise':
                    rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) * 3
                    grad.append( item.grad + rnd)  
                
                elif self.attack_type == 'filtering-attack':
                    grad.append(item.grad)
                    # refer to agent.py to see attack
                
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
