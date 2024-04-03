"""
A simple MLP implemented in PyTorch.
This model is an image encoder.
The input is a batch of images of dimension [batch_size, C, H, W].
The output is a sequence of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import torch.nn as nn
from einops.layers.torch import Rearrange

# must be called "Block" but this is a simple MLP
class Block(nn.Module):
    def __init__(self, img_size, num_tokens, token_dim):
        super(Block, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(img_size * img_size * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_tokens * token_dim),
            Rearrange('b (n d) -> b n d', n=num_tokens, d=token_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x