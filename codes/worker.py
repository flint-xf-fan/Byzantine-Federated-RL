import torch
import numpy as np
import gym
from gym.spaces import Discrete, Box
from policy import Policy
from utils import get_inner_model

class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity'
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."   
        
        # make policy network
        obs_dim = self.env.observation_space.shape[0]
        n_acts = self.env.action_space.n
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
        self.logits_net = Policy(self.sizes, activation, output_activation)
        
    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})
    
    def rollout(self, device, max_epi = 5000, render = False):
        env = gym.make(self.env_name)
        obs = env.reset()
        done = False  
        epi_ret = 0
        epi_len = 0
        for _ in range(max_epi):
            if render:
                env.render()
            
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device))[0]
            obs, rew, done, _ = env.step(action)
            epi_len += 1
            epi_ret += rew
            if done:
                break
        return epi_ret, epi_len
    
    def collect_experience_for_training(self, B, device, record = False):
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
            act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, done, _ = self.env.step(act)
            # if self.env.unwrapped.spec.id == 'MountainCar-v0':
            #     if obs[0] >= 0.4:
            #         rew += 1
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)
            # save trajectory
            if record:
                batch_actions.append(act)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                for r in ep_rews[::-1]:
                    r /= 100
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
                batch_weights += returns

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break

        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)
        if record:
            batch_states = batch_states
            batch_actions = batch_actions
        
        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(B, device)
        
        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()
        
        # determine if the agent is byzantine
        if self.is_Byzantine:
            # return wrong gradient with noise
            grad = []
            for item in self.parameters():
                rnd = torch.rand(item.grad.shape, device = item.device) * (item.grad.max().data - item.grad.min().data) + item.grad.min().data
                rnd2 = ((torch.rand(item.grad.shape, device = item.device) > 0.5).float())
                # grad.append(item.grad + rnd * rnd2 )    
                grad.append(rnd * 2 * rnd2)    
    
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