import torch
import numpy as np

from torch.nn import DataParallel

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def env_wrapper(name, obs):
    if name == 'parking-v0':
        KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']
        return np.concatenate([obs[key] for key in KEY_ORDER])
    return obs