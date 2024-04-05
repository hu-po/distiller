"""
This model is an image encoder implemented in JAX.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import jax
from flax import linen as nn
import jax.numpy as jnp

class Model(nn.Module):
    num_tokens: int
    token_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x has shape [batch_size, H, W, C]
        x = nn.Conv(features=self.token_dim, kernel_size=(14, 14), strides=(14, 14))(x)
        x = x.reshape((x.shape[0], -1, self.token_dim))  # [batch_size, num_patches, token_dim]
        positions = jnp.arange(0, x.shape[1])
        position_embeddings = nn.Embed(num_embeddings=x.shape[1], features=self.token_dim)(positions)
        x += position_embeddings
        x = nn.LayerNorm()(x)
        attention_output = nn.SelfAttention(num_heads=2, qkv_features=self.token_dim, use_bias=False)(x)
        x += attention_output
        x = nn.LayerNorm()(x)
        dense_output = nn.Dense(features=self.token_dim)(x)
        print(f"dense_output shape: {dense_output.shape}")
        x = nn.relu(x)
        x += dense_output
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.num_tokens * self.token_dim)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], self.num_tokens, self.token_dim))
        # output has shape [batch_size, num_tokens, token_dim]
        return x

def make_model(num_tokens: int, token_dim: int):

    def init_params(rng_key, img_size):
        model = Model(num_tokens=num_tokens, token_dim=token_dim)
        dummy_input = jnp.zeros((1, img_size, img_size, 3))
        params = model.init(rng_key, dummy_input)
        return params

    def predict(x, params):
        # x has shape [batch_size, H, W, C]
        model = Model(num_tokens=num_tokens, token_dim=token_dim)
        return model.apply(params, x)

    return init_params, predict
