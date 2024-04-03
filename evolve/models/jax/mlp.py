"""
A simple MLP implemented in JAX.
This model is an image encoder.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a sequence of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Flatten, Identity


def make_encoder(num_tokens: int, token_dim: int):
    init_params, predict = stax.serial(
        Flatten,
        Dense(1024),
        Relu,
        Dense(1024),
        Relu,
        Dense(num_tokens * token_dim),
        stax.Reshape((num_tokens, token_dim)),
        Identity,
    )
    return init_params, predict
