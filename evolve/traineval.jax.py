import argparse
import time
import itertools
import os
import tqdm

import jax
import jax.numpy as jnp
import pandas as pd
from PIL import Image

from jax import jit, grad, random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test.jax")
parser.add_argument("--round", type=int, default=0)

parser.add_argument("--train_data_dir", type=str, default="/data/train")
parser.add_argument("--test_data_dir", type=str, default="/data/test")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--logs_dir", type=str, default="/logs")

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--img_mu", type=str, default="0.485,0.456,0.406")
parser.add_argument("--img_std", type=str, default="0.229,0.224,0.225")

parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--early_stop", type=int, default=6)

parser.add_argument("--max_model_size", type=int, default=1e8)
parser.add_argument("--output_seq_len", type=int, default=64)
parser.add_argument("--output_hidden_dim", type=int, default=32)

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

# ---- Verify Model

# when running in the docker container the model will be in /src/model.py
if os.path.exists("/src/model.py"):
    from model import init_params, predict
else:
    # for local testing use hardcoded path
    from evolve.models.jax.mlp import init_params, predict

# verify output shape
mock_img = jnp.zeros((args.batch_size, args.img_size, args.img_size, 3))
params = init_params(rng)
params["output_seq_len"] = args.output_seq_len
params["output_hidden_dim"] = args.output_hidden_dim
output = predict(mock_img, params)
assert output.shape == (args.batch_size, args.output_seq_len, args.output_hidden_dim), f"Invalid output shape: {output.shape}"

# verify model size
model_size = sum(param.size for param in jax.tree_flatten(params)[0])
assert model_size < args.max_model_size, f"Model size {model_size} exceeds max model size {args.max_model_size}"
print(f"Model size: {model_size}")
hparams["model_size"] = model_size

# ---- Dataset

distill_targets = [
    ("clip-vit-base-patch16", ()),
    ("dinov2-small", ()),
    ("siglip-base-patch16-224", ()),
    ("clip-vit-large-patch14-336", ()),
    ("dinov2-giant", ()),
    ("siglip-large-patch16-384", ()),
]

assert os.path.exists(
    args.train_data_dir
), f"Training data directory {args.train_data_dir} does not exist."
assert os.path.exists(
    args.test_data_dir
), f"Testing data directory {args.test_data_dir} does not exist."

# normalization of images depends on dataset
image_mu = [float(mu) for mu in args.img_mu.split(",")]
image_std = [float(std) for std in args.img_std.split(",")]
assert len(image_mu) == len(image_std) == 3, "Invalid image mean and std"
assert all(0 <= mu <= 1 for mu in image_mu), "Invalid image mean"
assert all(0 <= std <= 1 for std in image_std), "Invalid image std"

def custom_dataset(csv_files, npy_files, img_dir):
    def parse_embeddings(filename):
        return jnp.array(np.load(filename))

    def parse_images(csv_file, img_dir):
        csv_data = pd.read_csv(csv_file)
        images = []
        for img_path in csv_data.iloc[:, 0]:
            img_path = os.path.join(img_dir, img_path)
            image = Image.open(img_path).convert("RGB")
            image = image.resize((args.img_size, args.img_size))
            image = np.array(image, dtype=np.float32)
            mean = np.array(image_mu, dtype=np.float32)
            std = np.array(image_std, dtype=np.float32)
            image = (image / 255.0 - mean) / std

            images.append(image)
        return jnp.array(images)

    embeddings_list = [parse_embeddings(npy_file) for npy_file in npy_files]
    images = parse_images(csv_files[0], img_dir)

    assert all(
        len(embeddings) == len(embeddings_list[0]) for embeddings in embeddings_list
    ), "Number of rows in NPY files do not match"
    assert len(images) == len(
        embeddings_list[0]
    ), "Number of images and embeddings do not match"

    return images, embeddings_list

# Load the custom dataset
train_images, train_embeddings_list = custom_dataset(
    csv_files=[f"{args.train_data_dir}/{t[0]}.csv" for t in distill_targets],
    npy_files=[f"{args.train_data_dir}/{t[0]}.npy" for t in distill_targets],
    img_dir=args.train_data_dir,
)
test_images, test_embeddings_list = custom_dataset(
    csv_files=[f"{args.test_data_dir}/{t[0]}.csv" for t in distill_targets],
    npy_files=[f"{args.test_data_dir}/{t[0]}.npy" for t in distill_targets],
    img_dir=args.test_data_dir,
)

num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, args.batch_size)
num_batches = num_complete_batches + bool(leftover)


def data_stream(rng):
    while True:
        perm = random.permutation(rng, num_train)
        for i in range(num_batches):
            batch_idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            yield train_images[batch_idx], [
                embeddings[batch_idx] for embeddings in train_embeddings_list
            ]


batches = data_stream(rng)

# ---- Full Model w/ Heads

params = {
    "backbone": init_params(rng),
    "head": {f"proj.{t[0]}": random.normal(rng, t[1]) for t in distill_targets},
}

def model_head(x, params, embedding_dim=(None, None)):
    pass

def model(x, params):
    x = predict(x, params["backbone"])
    outputs = []
    for t in distill_targets:
        x = model_head(x, params["head"][f"proj.{t[0]}"], embedding_dim=t[1])
        outputs.append(x)
    return outputs


# ----- Optimizer and Loss


def criterion(params, batch):
    images, labels = batch
    logits = model(images, params)
    return jnp.mean((logits - labels) ** 2)


opt_init, opt_update, get_params = optimizers.adam(args.learning_rate, args.b1, args.b2)


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
plt.plot(range(epoch), hist_loss_train, label="train")
plt.plot(range(epoch), hist_loss_test, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
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
