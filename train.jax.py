import argparse
import itertools
import os
import time

from PIL import Image
import jax
from jax import grad, jit, nn, random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--logs_dir", type=str, default="/logs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test.pytorch")
parser.add_argument("--round", type=int, default=0)
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--train_data_dir", type=str, default="sdxl_imagenet_8/train")
parser.add_argument("--test_data_dir", type=str, default="sdxl_imagenet_8/test")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--train_img_mu", type=str, default="0.558373,0.519655,0.478256")
parser.add_argument("--train_img_std", type=str, default="0.207305,0.191163,0.185902")
parser.add_argument("--test_img_mu", type=str, default="0.558373,0.519655,0.478256")
parser.add_argument("--test_img_std", type=str, default="0.207305,0.191163,0.185902")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--early_stop", type=int, default=2)
parser.add_argument("--max_model_size", type=int, default=int(1e8))
parser.add_argument("--num_tokens", type=int, default=8)
parser.add_argument("--token_dim", type=int, default=16)
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
    from model import make_model
else:
    # for local testing use hardcoded path
    from models.jax.cnn import make_model

# verify output shape
mock_img = jnp.zeros((args.batch_size, args.img_size, args.img_size, 3))
init_params, predict = make_model(args.num_tokens, args.token_dim)
params = init_params(rng, args.img_size)
output = predict(mock_img, params)
assert output.shape == (args.batch_size, args.num_tokens, args.token_dim), f"Invalid output shape: {output.shape}"

# verify model size
model_size = sum(param.size for param in jax.tree_util.tree_flatten(params)[0])
assert model_size < args.max_model_size, f"Model size {model_size} exceeds max model size {args.max_model_size}"
print(f"Model size: {model_size}")
hparams["model_size"] = model_size

# ---- Dataset

distill_targets = {
    "clip-vit-base-patch16" : (197, 768),
    "dinov2-small" : (257, 384),
    # "siglip-base-patch16-224" : (196, 768),
    # "clip-vit-large-patch14-336" : (577, 1024),
    # "dinov2-giant" : (257, 1536),
    # "siglip-large-patch16-384" : (576, 1024),
}

train_data_path = os.path.join(args.data_dir, args.train_data_dir)
assert os.path.exists(train_data_path), f"{args.train_data_dir} does not exist."
print(f"Training data directory: {train_data_path}")
test_data_path = os.path.join(args.data_dir, args.test_data_dir)
assert os.path.exists(test_data_path), f"{args.test_data_dir} does not exist."
print(f"Test data directory: {test_data_path}")

# normalization of images depends on dataset
train_img_mu = [float(mu) for mu in args.train_img_mu.split(",")]
train_img_std = [float(std) for std in args.train_img_mu.split(",")]
assert len(train_img_mu) == len(train_img_std) == 3, "Invalid image mean and std"
assert all(0 <= mu <= 1 for mu in train_img_mu), "Invalid image mean"
assert all(0 <= std <= 1 for std in train_img_std), "Invalid image std"
test_img_mu = [float(mu) for mu in args.test_img_mu.split(",")]
test_img_std = [float(std) for std in args.test_img_std.split(",")]
assert len(test_img_mu) == len(test_img_std) == 3, "Invalid image mean and std"
assert all(0 <= mu <= 1 for mu in test_img_mu), "Invalid image mean"
assert all(0 <= std <= 1 for std in test_img_std), "Invalid image std"

def custom_dataset(csv_files, npy_files, img_dir, img_mu, img_std):
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
            mean = np.array(img_mu, dtype=np.float32)
            std = np.array(img_std, dtype=np.float32)
            image = (image / 255.0 - mean) / std

            images.append(image)
        return jnp.array(images)

    targets = {k: parse_embeddings(npy_file) for k, npy_file in npy_files.items()}
    images = parse_images(list(csv_files.values())[0], img_dir)

    assert all(
        len(target) == len(targets[list(targets.keys())[0]]) for target in targets.values()
    ), "Number of rows in NPY files do not match"
    assert len(images) == len(
        targets[list(targets.keys())[0]]
    ), "Number of images and embeddings do not match"

    return images, targets

# Load the custom dataset
train_images, train_targets = custom_dataset(
    csv_files={k : f"{train_data_path}/{k}.csv" for k, v in distill_targets.items()},
    npy_files={k : f"{train_data_path}/{k}.npy" for k, v in distill_targets.items()},
    img_dir=train_data_path,
    img_mu=train_img_mu,
    img_std=train_img_std,
)
test_images, test_targets = custom_dataset(
    csv_files={k : f"{test_data_path}/{k}.csv" for k, v in distill_targets.items()},
    npy_files={k : f"{test_data_path}/{k}.npy" for k, v in distill_targets.items()},
    img_dir=test_data_path,
    img_mu=test_img_mu,
    img_std=test_img_std,
)

num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, args.batch_size)
num_batches = num_complete_batches + bool(leftover)


def data_stream(rng):
    while True:
        perm = random.permutation(rng, num_train)
        for i in range(num_batches):
            batch_idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            yield train_images[batch_idx], {k : v[batch_idx] for k, v in train_targets.items()}


batches = data_stream(rng)

# ---- Full Model w/ Heads

init_params, predict = make_model(args.num_tokens, args.token_dim)
params = {"encoder": init_params(rng, args.img_size)}

params["heads"] = {}
for name, (num_tokens, token_dim) in distill_targets.items():
    params["heads"][name] = random.normal(rng, (args.token_dim, num_tokens * token_dim))

def model_head(x, head_params, num_tokens, token_dim):
    x = nn.relu(x @ head_params)
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1, token_dim)  # Reshape to (batch_size, num_tokens, token_dim)
    if x.shape[1] != num_tokens:
        # Adjust the number of tokens if there is a mismatch
        if x.shape[1] > num_tokens:
            x = x[:, :num_tokens, :]
        else:
            padding = jnp.zeros((batch_size, num_tokens - x.shape[1], token_dim))
            x = jnp.concatenate([x, padding], axis=1)
    return x

@jit
def model(x, params):
    x = predict(x, params["encoder"])
    outputs = {}
    for name, (num_tokens, token_dim) in distill_targets.items():
        head_params = params["heads"][name]
        x_head = model_head(x, head_params, num_tokens, token_dim)
        outputs[name] = x_head
    return outputs


# ----- Loss Function

@jit
def distillation_loss(params, batch):
    images, targets = batch
    outputs = model(images, params)
    loss = 0.0
    for name, output in outputs.items():
        target = targets[name]
        num_tokens_out, token_dim_out = output.shape[1], output.shape[2]
        num_tokens_target, token_dim_target = target.shape[1], target.shape[2]
        assert token_dim_out == token_dim_target, f"Hidden dimensions do not match for {name}: {token_dim_out} != {token_dim_target}"
        if num_tokens_out > num_tokens_target:
            output = output[:, :num_tokens_target, :]
        elif num_tokens_out < num_tokens_target:
            padding = jnp.zeros((output.shape[0], num_tokens_target - num_tokens_out, token_dim_out))
            output = jnp.concatenate([output, padding], axis=1)
        loss += jnp.mean((output - target) ** 2)
    
    return loss / len(outputs)


# ----- Optimizer and Loss

opt_init, opt_update, get_params = optimizers.adam(args.learning_rate, args.b1, args.b2)

@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(distillation_loss)(params, batch), opt_state)

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
    loss_train = distillation_loss(params, (train_images, train_targets))
    loss_train = float(jnp.asarray(loss_train))
    print(f"\t loss/train: {loss_train}")
    writer.add_scalar("loss/train", loss_train, epoch)

    # EVAL LOSS
    loss_test = distillation_loss(params, (test_images, test_targets))
    loss_test = float(jnp.asarray(loss_train))
    print(f"\t loss/test: {loss_test}")
    hist_loss_test.append(loss_test)
    writer.add_scalar("loss/test", loss_test, epoch)

    # early stopping
    if loss_test < best_loss:
        best_loss = loss_test
        last_best_epoch = epoch
    if epoch - last_best_epoch > args.early_stop:
        print(f"early stopping at epoch {epoch}")
        break

# Optionally save trained models to checkpoint
if args.save_ckpt:
    model_save_path = os.path.join(args.ckpt_dir, f"{args.run_name}.e{epoch}.pth")
    print(f"Saving model to {model_save_path}")
    # TODO

# Save plot of training and test accuracy for VLMs to analyze
plt.plot(range(len(hist_loss_train)), hist_loss_train, label="train")
plt.plot(range(len(hist_loss_test)), hist_loss_test, label="test")
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