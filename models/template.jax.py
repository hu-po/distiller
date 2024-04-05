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
        # TODO: Implement model architecture here
        # output has shape [batch_size, num_tokens, token_dim]
        return x


def make_model(num_tokens: int, token_dim: int):

    def init_params(rng_key, img_size):
        block = Model(num_tokens=num_tokens, token_dim=token_dim)
        dummy_input = jnp.zeros((1, img_size, img_size, 3))
        params = block.init(rng_key, dummy_input)
        return params

    def predict(x, params):
        block = Model(num_tokens=num_tokens, token_dim=token_dim)
        return block.apply(params, x)

    return init_params, predict
