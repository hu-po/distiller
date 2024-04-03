"""
A simple CNN implemented in JAX.
This model is an image encoder.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a sequence of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import random


def make_encoder(num_tokens: int, token_dim: int):

    def predict(x, params):
        conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b = params

        # First convolutional layer
        x = jnp.transpose(x, (0, 3, 1, 2))  # Reshape to [batch_size, C, H, W]
        x = nn.conv(x, conv1_w, conv1_b, (1, 1), 'SAME')
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')

        # Second convolutional layer
        x = nn.conv(x, conv2_w, conv2_b, (1, 1), 'SAME')
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')

        # Flatten the output
        x = x.reshape(x.shape[0], -1)

        # Fully connected layer
        x = jnp.dot(x, fc_w) + fc_b
        x = nn.relu(x)

        # Reshape to [batch_size, num_tokens, token_dim]
        x = x.reshape(x.shape[0], num_tokens, token_dim)
        return x

    def init_params(rng_key, img_size):
        conv1_key, conv2_key, fc_key = random.split(rng_key, 3)

        # First convolutional layer parameters
        conv1_w = random.normal(conv1_key, (3, 3, 3, 32)) * 0.01
        conv1_b = jnp.zeros((32,))

        # Second convolutional layer parameters
        conv2_w = random.normal(conv2_key, (3, 3, 32, 64)) * 0.01
        conv2_b = jnp.zeros((64,))

        # Fully connected layer parameters
        output_dim = num_tokens * token_dim
        fc_input_dim = 64 * (img_size // 4) * (img_size // 4)
        fc_w = random.normal(fc_key, (fc_input_dim, output_dim)) * 0.01
        fc_b = jnp.zeros((output_dim,))

        return (conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b)

    return init_params, predict