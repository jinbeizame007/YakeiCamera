import torch
import random

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class ReplayBuffer():
    def __init__(self):
        self.buffer_size = 50
        self.buffer = [None for i in range(self.buffer_size)]
        self.index = 0
        self.max = 0

    def add(self, data):
        self.buffer[self.index] = data
        self.index += 1
        self.max += 1
        self.index %= 50
    
    def sample(self):
        return self.buffer[random.randint(0, min(self.max-1, self.buffer_size-1))]