import torch
import numpy as np

from torch.nn import DataParallel

from matplotlib import animation
import matplotlib.pyplot as plt

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
    elif name == 'highway-v0':
        return np.transpose(np.array(obs, np.float32)).reshape(1,4,84,84)
    else:
        return obs

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=120)
    