import torch
import numpy as np
import gym
from gym.spaces import Discrete, Box
from policy import Policy
from utils import get_inner_model
from copy import deepcopy
import math

class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 beam_num,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = 'None'
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        self.beam_num = beam_num
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.attack_type = attack_type
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
    
    def rollout(self, device, max_epi = 5000, render = False, env = None, obs = None):
        
        if env is None and obs is None:
            env = gym.make(self.env_name)
            obs = env.reset()
        done = False  
        ep_rew = []
        for _ in range(max_epi):
            if render:
                env.render()
            
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device))[0]
            obs, rew, done, _ = env.step(action)
            ep_rew.append(rew)
            if done:
                break
            
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        seed = np.random.randint(0, 10000)
        self.env.seed(seed)
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        batch_states = []
        batch_actions = []

        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            batch_states.append(obs)
            # act in the environment
            act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, done, _ = self.env.step(act)
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)
            # save trajectory
            batch_actions.append(act)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # when to do sampling
                num_beam_start_states = math.ceil(math.log(ep_len, 3))
                beam_start_states = sorted(list(set([(ep_len - 3**i) for i in range(2, num_beam_start_states)] + [0])))
	
                # run baseline
                baseline = []
            	
                # revisit the same trajectory as sampled above and get baseline value
                self.env.seed(seed)
                self.env.reset()
                for i in range(ep_len):
                    if i in beam_start_states:
                        beam_state = beam_start_states.index(i)
                        beam_freq = (beam_start_states[beam_state + 1] if (beam_state + 1) < len(beam_start_states) else (ep_len)) - i
                        beams = [self.rollout(device = device, render = False, env = deepcopy(self.env), obs = obs)[-1] for _ in range(self.beam_num)]                
                        beams_returns = []
                        for b in beams:
                            returns = []
                            R = 0
                            for r in b[::-1]:
                                R = r + self.gamma * R
                                returns.insert(0, R)
                            beams_returns.append(returns)
                        beams_returns = [sum([Gbr[baseline_step] if len(Gbr) > baseline_step else Gbr[-1] for Gbr in beams_returns]) / len(beams_returns) for baseline_step in range(beam_freq)]
                        baseline += beams_returns
                    obs, _, _, _ = self.env.step(batch_actions[-ep_len:][i])

                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns) - torch.tensor(baseline)
                batch_weights += returns

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                seed = np.random.randint(0, 10000)
                self.env.seed(seed)
                obs, done, ep_rews = self.env.reset(), False, []



        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

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
                # rnd_11 = (torch.rand(item.grad.shape, device = item.device) * 2. - 1.)
                # rnd_o = ((torch.rand(item.grad.shape, device = item.device) > 0.5).float())
                # grad.append(item.grad + item.grad * rnd_11 * rnd_o * 2)  
                
                item.grad[item.grad > item.grad.mean()] = -item.grad[item.grad > item.grad.mean()] * 2
                grad.append(item.grad)  
    
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