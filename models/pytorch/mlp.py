"""
This model is an image encoder implemented in PyTorch.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import torch.nn as nn
from einops.layers.torch import Rearrange

class Model(nn.Module):
    def __init__(self, img_size, num_tokens, token_dim):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(img_size * img_size * 3, num_tokens * token_dim),
            nn.ReLU(),
            Rearrange('b (n d) -> b n d', n=num_tokens, d=token_dim)
        )

    def forward(self, x):
        # x has shape [batch_size, C, H, W]
        x = self.flatten(x)
        x = self.layers(x)
        # output has shape [batch_size, num_tokens, token_dim]
        return x