"""
This model is an image encoder implemented in PyTorch.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, img_size: int, num_tokens:int, token_dim:int):
        super(Model, self).__init__()
        # TODO: Initialize layers here

    def forward(self, x):
        # x has shape [batch_size, C, H, W]
        # TODO: Implement forward pass
        # output has shape [batch_size, num_tokens, token_dim]
        return x