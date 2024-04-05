"""
This model is an image encoder implemented in JAX.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import jax.numpy as jnp
import jax.nn as nn
from jax import random


def make_model(num_tokens: int, token_dim: int):

    def init_params(rng_key, img_size):
        input_dim = img_size * img_size * 3
        output_dim = num_tokens * token_dim
        w_key, b_key = random.split(rng_key)
        w = random.normal(w_key, (input_dim, output_dim)) * 0.01
        b = random.normal(b_key, (output_dim,)) * 0.01
        return (w, b)

    def predict(x, params):
        # x has shape [batch_size, H, W, C]
        w, b = params
        x = x.reshape(x.shape[0], -1)
        x = jnp.dot(x, w) + b
        x = nn.relu(x)
        x = x.reshape(x.shape[0], num_tokens, token_dim)
        # output has shape [batch_size, num_tokens, token_dim]
        return x

    return init_params, predict
