
import random
import torch
import torch.nn.functional as F
from collections import namedtuple, deque



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_valid_mask'))

# Replay Memory class to store past experiences for experience replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    


# In 2048, the state is represented as a 4x4 grid of integers, where each integer represents the exponent of 2 for the tile value (0 for empty, 1 for 2, 2 for 4, ..., up to 15 for 32768). 
# To feed this into a neural network, we can convert it into a one-hot encoded tensor with 16 channels (one for each possible tile value from 0 to 15). 
# This way, the network can learn separate features for each tile value.
def process_state(obs_np, device):
    """Convert the 4x4 NumPy board directly into a One-Hot Tensor on the GPU."""
    t = torch.tensor(obs_np, dtype=torch.long, device=device).clamp(max=15)
    t_oh = F.one_hot(t, num_classes=16).permute(2, 0, 1).unsqueeze(0).float()
    return t_oh