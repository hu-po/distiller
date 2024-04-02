import torch.nn as nn
from einops.layers.torch import Rearrange

# must be called "Block" but this is a simple MLP
class Block(nn.Module):
    def __init__(self, num_classes=None):
        super(Block, self).__init__()
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.layers = nn.Sequential(
            nn.Linear(224*224*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x