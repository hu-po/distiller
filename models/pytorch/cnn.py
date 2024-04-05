"""
A simple CNN implemented in PyTorch.
This model is an image encoder.
The input is a batch of images of dimension [batch_size, C, H, W].
The output is a sequence of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import torch.nn as nn
from einops.layers.torch import Rearrange

class Model(nn.Module):
    def __init__(self, img_size, num_tokens, token_dim):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(64 * (img_size // 8) * (img_size // 8), 64),
            nn.ReLU(),
            nn.Linear(64, num_tokens * token_dim),
            Rearrange('b (n d) -> b n d', n=num_tokens, d=token_dim)
        )

    def forward(self, x):
        # x has shape [batch_size, C, H, W]
        x = self.conv(x)
        x = self.flatten(x)
        x = self.layers(x)
        # output has shape [batch_size, num_tokens, token_dim]
        return x