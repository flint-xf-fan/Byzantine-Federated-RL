import torch
from torch import nn
import gym
import numpy as np
import torch.optim as optim
from gym.spaces import Discrete, Box
from torch.distributions.categorical import Categorical
from utils import get_inner_model

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def __init__(self,
                 id,
                 env_name,
                 hidden_units,
                 activation = 'Tanh',
                 output_activation = 'Identity'
                 ):
        super(Actor, self).__init__()
        
        # store parameters
        self.id = id
        self.activation = activation
        self.output_activation = output_activation
        
        if activation == 'Tanh':
            self.activation = nn.Tanh
        else:
            raise NotImplementedError
            
        if output_activation == 'Identity':
            self.output_activation = nn.Identity
        else:
            raise NotImplementedError
        
        # make environment, check spaces, get obs / act dims
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
        self.logits_net = mlp(self.sizes, self.activation, self.output_activation)
        
        # make optimizer
        self.optimizer = optim.Adam(self.logits_net.parameters())

    def forward(self, obs, sample = True):
        """
        :param input: (obs) input observation
        :return: action
        """
        
        logits = self.logits_net(obs)
        policy = Categorical(logits=logits)
        
        if sample:
            action = policy.sample()
        else:
            action = policy.probs.argmax()
        
        return action.item(), policy.log_prob(action)
    
    def load_net_param(self, param):
        model_actor = get_inner_model(self)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})
    
    def train_one_epoch(self, batch_size, device):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:

            # act in the environment
            act, log_prob = self(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, done, _ = self.env.step(act)
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_log_prob) > batch_size:
                    break

        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        # return gradient        
        return [item.grad for item in self.parameters()], batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens) 
