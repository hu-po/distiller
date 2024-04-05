"""
This model is an image encoder implemented in JAX.
The input is a batch of images of dimension [batch_size, H, W, C].
The output is a batch of sequences of image tokens of dimension [batch_size, num_tokens, token_dim].
"""

from flax import linen as nn
import jax.numpy as jnp


class Model(nn.Module):
    num_tokens: int
    token_dim: int

    @nn.compact
    def __call__(self, x):
        # x has shape [batch_size, H, W, C]
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.num_tokens * self.token_dim)(x)
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
