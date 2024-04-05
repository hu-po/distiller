"""
This model is an image encoder implemented in PyTorch.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, img_size: int, num_tokens:int, token_dim:int):
        super(Model, self).__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.patch_conv = nn.Conv2d(in_channels=3, out_channels=token_dim, kernel_size=14, stride=14)
        num_patches = (img_size // 14) * (img_size // 14)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, token_dim))
        self.layer_norm1 = nn.LayerNorm(token_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=2, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(token_dim)
        self.dense = nn.Linear(in_features=token_dim, out_features=token_dim)
        self.final_dense = nn.Linear(in_features=num_patches * token_dim, out_features=num_tokens * token_dim)

    def forward(self, x):
        # x has shape [batch_size, C, H, W]
        x = self.patch_conv(x)
        x = x.flatten(2)  # Flatten the patches while keeping the batch and patch dimension separate.
        x = x.permute(0, 2, 1)  # Convert [batch_size, C, num_patches] to [batch_size, num_patches, C]
        x = x + self.position_embeddings[:, :x.shape[1], :]
        x = self.layer_norm1(x)
        attention_output, _ = self.self_attention(x, x, x)
        x = x + attention_output
        x = self.layer_norm2(x)
        dense_output = self.dense(x)
        x = x + F.relu(dense_output)
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.final_dense(x))
        x = x.reshape((x.shape[0], self.num_tokens, self.token_dim))
        # output has shape [batch_size, num_tokens, token_dim]
        return x
