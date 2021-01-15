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
from highway_env import __init__
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
        
        if opts.highway:
            
            if env_name == 'parking-v0':
                self.env.configure({
                    'vehicles_count': 1,
                    "simulation_frequency": 10,
                    "policy_frequency": 2,
                    "duration": 100,
                    "collision_reward" : -10,
                    "reward_speed_range": [20, 40]
                })
            elif env_name == 'highway-v0':
                pass
            else:
                self.env.configure({
                        "observation": {
                            "type": "Kinematics",
                            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                            "features_range": {
                                        "x": [-100, 100],
                                        "y": [-100, 100],
                                        "vx": [-20, 20],
                                        "vy": [-20, 20]
                                    },
                            "absolute": False,
                            "order": "sorted",
                        },
                        'vehicles_count': 10,
                        "simulation_frequency": 10,
                        "policy_frequency": 2,
                        "duration": 100,
                        "collision_reward" : -10,
                        "reward_speed_range": [20, 40]
                    })
            if id==1: 
                print('Env configuration for highway-env:',)
                pprint.pprint(self.env.config)
            obs = self.env.reset()
                
            if opts.discrete:
                
                obs_dim = obs.size
                n_acts = self.env.action_space.n
                
                hidden_sizes = list(eval(hidden_units))
                self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
                
                if env_name == 'highway-v0':
                    self.logits_net = CnnPolicy(self.sizes, activation, output_activation)
                else:
                    self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
                
            else:
                
                obs = env_wrapper(opts.env_name, obs)                
                obs_dim = obs.size
                n_acts = self.env.action_space.shape[0]
            
                hidden_sizes = list(eval(hidden_units))
                self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
                
                self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation)

        
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
                    if id==1: print('geer = ', 0.4)
                    self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation, geer = 0.4)
        
                elif self.env_name == 'Pendulum-v0':
                    if id==1: print('geer = ', 2)
                    self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation, geer = 2)
                    
                else:
                    self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        ################
        self.use_critic = opts.use_critic
        if self.use_critic:
            self.critic = LinearCritic(self.sizes)
        ################
        
        if self.id == 1:
            print(self.logits_net)

    
    def load_param_from_master(self, param, param_for_critic):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})
        
        if self.use_critic:
            self.critic.set_parameters(param_for_critic)
    
    def rollout(self, device, render = False, env = None, obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = self.env
            self.config()
            obs = env.reset()
        else:
            self.config()
        done = False  
        ep_rew = []
        frames = []
        while not done:
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
    
    def config(self):
        if self.env_name == 'highway-v0':
            screen_width, screen_height = 84, 84
            config = {
                "offscreen_rendering": True,
                "observation": {
                    "type": "GrayscaleObservation",
                    "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                    "stack_size": 4,
                    "observation_shape": (screen_width, screen_height)
                },
                "screen_width": screen_width,
                "screen_height": screen_height,
                # "scaling": 1.75,
                "policy_frequency": 2
            }
            self.env.configure(config)
        
    
    def collect_experience_for_training(self, B, device, record = False, sample = True, critic_loss = False, epsilon = 0.2):
        self.config()
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
        
        #########
        if self.use_critic:
            critic_value = []
            if critic_loss:
                self.input_value = []
                self.output_value = []
        #########

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            if np.random.rand() < epsilon:
                rnd_action = self.env.action_space.sample()
            else:
                rnd_action = None
            
            act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = rnd_action)
            #########
            if self.use_critic:
                in_, out_ = self.critic.predict(torch.as_tensor(obs, dtype=torch.float32).to(device), t)
                critic_value.append(out_)
                if critic_loss:
                    self.input_value.append(in_)
            #########
            obs, rew, done, info = self.env.step(act)
            t = t + 1
            
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
                #########
                if not done and self.use_critic:
                    R = self.critic.predict(torch.as_tensor(obs, dtype=torch.float32).to(device), t)[1]
                else:
                    R = 0
                #########
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                #print(returns)
                returns = torch.tensor(returns, dtype=torch.float32)

                #########
                if self.use_critic:
                    critic_value_detach = torch.as_tensor(critic_value).view(-1)
                        
                    advantage = returns - critic_value_detach
                    # print(advantage)
                    # print(advantage)

                    critic_value = []
                    if critic_loss: self.output_value += returns.tolist()
                    
                #########
                else:
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
