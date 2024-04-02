import argparse
import time
import itertools
import os
import tqdm

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

# ---- Dataset

assert os.path.exists(args.train_data_dir), f"Training data directory {args.train_data_dir} does not exist."
assert os.path.exists(args.test_data_dir), f"Testing data directory {args.test_data_dir} does not exist."

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

# ---- Model

params = {
    "backbone" : init_params(rng),
    "head" : {
        "proj.clip" : random.normal(rng, (dim_seq, num_classes)),
        "proj.dino" : random.normal(rng, (dim_seq, num_classes)),
        "proj.siglip" : random.normal(rng, (dim_seq, num_classes)),
    }
}

def model(x, params):
    x = predict(x, params["backbone"])
    # TODO: model head for each embedding to be distilled
    return x

# ----- Optimizer and Loss

def criterion(params, batch):
    images, labels = batch
    logits = model(images, params)
    return jnp.mean((logits - labels) ** 2)

opt_init, opt_update, get_params = optimizers.adam(
    args.learning_rate, args.b1, args.b2
)

@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(criterion)(params, batch), opt_state)

opt_state = opt_init(params)
itercount = itertools.count()

# ---- Training Loop

print("Starting training...")
hist_loss_train = []
hist_loss_test = []
best_loss = np.inf
last_best_epoch = 0
for epoch in range(args.num_epochs):
    epoch_start_time = time.time()

    for batch_idx in tqdm(range(num_batches), desc=f"train.epoch.{epoch}"):
        opt_state = update(next(itercount), opt_state, next(batches))
        print(f"Batch {batch_idx+1}/{num_batches}", end="\r")

    epoch_duration = time.time() - epoch_start_time
    print(f"\t duration {epoch_duration:0.2f} sec")

    # TRAIN LOSS
    params = get_params(opt_state)
    loss_train = criterion(params, (train_images, train_labels))
    print(f"\t loss/train: {loss_train}")
    writer.add_scalar("loss/train", loss_train, epoch)
    if loss_train < best_loss:
        best_loss = loss_train
        last_best_epoch = epoch

    # EVAL LOSS
    loss_test = criterion(params, (test_images, test_labels))
    print(f"\t loss/test: {loss_test}")
    hist_loss_test.append(loss_test)
    writer.add_scalar("loss/test", loss_test, epoch)
    
    # early stopping
    if epoch - last_best_epoch > args.early_stop:
        print(f"early stopping at epoch {epoch}")
        break

# Optionally save trained models to checkpoint
if args.save_ckpt:
    model_save_path = os.path.join(args.ckpt_dir, f"{args.run_name}.e{epoch}.pth")
    print(f"Saving model to {model_save_path}")
    # TODO

# Save plot of training and test accuracy for VLMs to analyze
plt.plot(range(epoch), hist_loss_train, label='train')
plt.plot(range(epoch), hist_loss_test, label='test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f"{args.ckpt_dir}/plot.{args.run_name}.png")

# Update tensorboard logger and results yaml
scores = {"loss.test": loss_test}
writer.add_hparams(hparams, scores)
writer.close()
results_filepath = os.path.join(args.ckpt_dir, f"results.r{args.round}.yaml")
if os.path.exists(results_filepath):
    with open(results_filepath, "r") as f:
        results = yaml.safe_load(f) or {}
else:
    results = {}
hparams.update(scores)
print(f"Writing results to {results_filepath}")
results[args.run_name] = hparams
with open(results_filepath, "w") as f:
    yaml.dump(results, f)