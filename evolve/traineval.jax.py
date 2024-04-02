import argparse
import time
import itertools
import os

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
import numpy.random as npr
from os import path
import numpy as np
import struct
import yaml
from tensorboardX import SummaryWriter

# When running in the docker container the model will be in /src/model.py
if os.path.exists("/src/model.py"):
    from model import init_params, predict
else:
    # for local testing use hardcoded path
    from evolve.models.jax.mlp import init_params, predict

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test.jax")
parser.add_argument("--round", type=int, default=0)

parser.add_argument("--train_data_dir", type=str, default="/data/train")
parser.add_argument("--test_data_dir", type=str, default="/data/test")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--logs_dir", type=str, default="/logs")

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--early_stop", type=int, default=6)

parser.add_argument("--max_model_size", type=int, default=1e8)

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.95)

args = parser.parse_args()

# Check if GPU is available
print("jax.devices:", jax.devices())
print("jax.default_backend():", jax.default_backend())

# Set up tensorboard logging
tblog_filepath = os.path.join(args.logs_dir, args.run_name)
writer = SummaryWriter(tblog_filepath)
print(f"Writing logs to {tblog_filepath}")
hparams = {
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
}

# Set the random seed
rng = random.PRNGKey(args.seed)

def mnist_raw(data_dir: str):
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename, data_dir)

    train_images = parse_images(path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False, data_dir:str="/data/mnist"):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw(data_dir)

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels



def model(x, params):
    # model is a stack of mamba blocks
    for block_params in params["residual_blocks"]:
        y = mamba_block(x, block_params["mamba_params"])
        # apply rms norm after each block
        y = rms_norm(y, block_params["norm_w"], block_params["norm_b"])
        # skip connection
        x += y
    # remove channel dimmension
    x = jnp.squeeze(x, axis=-1)
    # classification head
    logits = x @ params["class_head_w"] + params["class_head_b"]
    return logits


def cross_entropy_loss(params, batch):
    """Computes cross-entropy loss for Mamba model on MNIST."""
    images, labels = batch
    logits = model(images, params)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * labels, axis=-1))


def accuracy(params, batch):
    """Computes accuracy for Mamba model on MNIST."""
    images, labels = batch
    images = jnp.expand_dims(images, axis=-1)
    predicted_labels = jnp.argmax(model(images, params), axis=-1)
    return jnp.mean(predicted_labels == jnp.argmax(labels, axis=-1))


# Load MNIST dataset
train_images, train_labels, test_images, test_labels = mnist(data_dir=args.data_dir)
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, args.batch_size)
num_batches = num_complete_batches + bool(leftover)
num_classes = 10
dim_c = 1
dim_seq = 28 * 28

def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            yield jnp.expand_dims(train_images[batch_idx], axis=-1), train_labels[
                batch_idx
            ]

batches = data_stream()

params = {
    "residual_blocks": [
        {
            "mamba_params": {
                # input projection
                "in_proj_w": random.normal(rng, (dim_c, args.dim_h)),
                "in_proj_b": jnp.zeros(args.dim_h),
                # SSM parameters
                "B_proj_w": random.normal(rng, (args.dim_h, args.dim_h)),
                "C_proj_w": random.normal(rng, (args.dim_h, args.dim_h)),
                # Initialization for Δ from paper in Section 3.6
                "Δ_proj_w": random.uniform(
                    rng, (args.dim_h, args.dim_Δ), minval=0.001, maxval=0.1
                ),
                # Initialization for A from paper in Section 3.6 and Table 8
                "A": jnp.eye(args.dim_h)
                + jnp.tril(jnp.ones((args.dim_h, args.dim_h)), -1),
                # causal 1D convolution layer
                "conv_w": random.normal(
                    rng, (args.dim_conv, args.dim_h, args.dim_h)
                ),
                # initial hidden state h_0
                "h_0": random.normal(rng, (args.dim_h,)),
                # output projection
                "out_proj_w": random.normal(rng, (args.dim_h, dim_c)),
                "out_proj_b": jnp.zeros(dim_c),
            },
            # RMS normalization layer
            "norm_w": jnp.ones(dim_c),
            "norm_b": jnp.zeros(dim_c),
        }
        for _ in range(args.num_blocks)
    ],
    # classification head
    "class_head_w": random.normal(rng, (dim_seq, num_classes)),
    "class_head_b": jnp.zeros(num_classes),
}

# Optimizer
opt_init, opt_update, get_params = optimizers.adam(
    args.learning_rate, args.b1, args.b2
)

@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(cross_entropy_loss)(params, batch), opt_state)

opt_state = opt_init(params)
itercount = itertools.count()

print("\nStarting training...")
train_acc_history = []
test_acc_history = []
for epoch in range(args.num_epochs):
    start_time = time.time()
    for batch_idx in range(num_batches):
        opt_state = update(next(itercount), opt_state, next(batches))
        print(f"Batch {batch_idx+1}/{num_batches}", end="\r")
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")

plt.plot(range(args.num_epochs), train_acc_history, label='train')
plt.plot(range(args.num_epochs), test_acc_history, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

# Saving the plot to a PNG file
plt.savefig(f"{args.log_dir}/plot.{args.run_name}.png")