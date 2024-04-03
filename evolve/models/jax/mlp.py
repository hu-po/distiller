"""
A simple MLP implemented in JAX.
This model is an image encoder.
The input is a batch of images of dimmension [batch_size, H, W, C].
The output is a sequence of image tokens of dimmension [batch_size, num_tokens, token_dim].
"""

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax

init_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)