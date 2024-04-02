from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax

init_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)