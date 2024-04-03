import argparse
import os
import time

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test.pytorch")
parser.add_argument("--round", type=int, default=0)

parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--logs_dir", type=str, default="/logs")

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

parser.add_argument("--max_model_size", type=int, default=1e8)
parser.add_argument("--num_tokens", type=int, default=64)
parser.add_argument("--token_dim", type=int, default=32)

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.95)

args = parser.parse_args()

# Check if GPU is available
if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# ---- Verify Model

# when running in the docker container the model will be in /src/model.py
if os.path.exists("/src/model.py"):
    from model import Block
else:
    # for local testing use hardcoded path
    from evolve.models.pytorch.mlp import Block

# verify output shape
model = Block(
    num_tokens = args.num_tokens,
    token_dim = args.num_tokens,
).to(device)
for param in model.parameters():
    assert param.sum() != 0, "Model parameter(s) not initialized properly."
mock_img = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
assert model(mock_img).shape == torch.Size(
    [args.batch_size, args.num_tokens, args.token_dim])

# verify model size
model_size = sum(p.numel() for p in model.parameters())
assert (
    model_size < args.max_model_size
), f"Model size {model_size} exceeds limit {args.max_model_size}"
print(f"Model size: {model_size / 1e6}M")
hparams["model_size"] = (sum(p.numel() for p in model.parameters()),)

# ---- Dataset

distill_targets = [
    ("clip-vit-base-patch16", (197, 768)),
    ("dinov2-small", (257, 384)),
    ("siglip-base-patch16-224", (196, 768)),
    ("clip-vit-large-patch14-336", (577, 1024)),
    ("dinov2-giant", (257, 1536)),
    ("siglip-large-patch16-384", (576, 1024)),
]

train_data_path = os.path.join(args.data_dir, args.train_data_dir)
assert os.path.exists(train_data_path), f"{args.train_data_dir} does not exist."
print(f"Training data directory: {train_data_path}")
test_data_path = os.path.join(args.data_dir, args.test_data_dir)
assert os.path.exists(test_data_path), f"{args.test_data_dir} does not exist."
print(f"Test data directory: {test_data_path}")

# normalization of images depends on dataset
train_img_mu = [float(mu) for mu in args.img_mu.split(",")]
train_img_std = [float(std) for std in args.img_std.split(",")]
assert len(train_img_mu) == len(train_img_std) == 3, "Invalid image mean and std"
assert all(0 <= mu <= 1 for mu in train_img_mu), "Invalid image mean"
assert all(0 <= std <= 1 for std in train_img_std), "Invalid image std"
test_img_mu = [float(mu) for mu in args.test_img_mu.split(",")]
test_img_std = [float(std) for std in args.test_img_std.split(",")]
assert len(test_img_mu) == len(test_img_std) == 3, "Invalid image mean and std"
assert all(0 <= mu <= 1 for mu in test_img_mu), "Invalid image mean"
assert all(0 <= std <= 1 for std in test_img_std), "Invalid image std"

class DistilDataset(Dataset):
    def __init__(self, csv_files, npy_files, img_dir, transform=None):
        self.csv_data = [pd.read_csv(csv_file) for csv_file in csv_files]
        self.embeddings = [np.load(npy_file) for npy_file in npy_files]
        self.img_dir = img_dir
        self.transform = transform
        assert all(
            len(csv_data) == len(self.csv_data[0]) for csv_data in self.csv_data
        ), "Number of rows in CSV files do not match"
        assert all(
            len(embeddings) == len(self.embeddings[0]) for embeddings in self.embeddings
        ), "Number of rows in NPY files do not match"

    def __len__(self):
        return len(self.csv_data[0])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.csv_data[0].iloc[idx, 0])
        embeddings = [
            torch.from_numpy(embeddings[idx]) for embeddings in self.embeddings
        ]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, embeddings

train_dataset = DistilDataset(
    csv_files=[f"{train_data_path}/{t[0]}.csv" for t in distill_targets],
    npy_files=[f"{train_data_path}/{t[0]}.npy" for t in distill_targets],
    img_dir=train_data_path,
    transform=transforms.Compose(
    [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_img_mu, std=train_img_std),
    ]
),
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
assert len(train_dataset) > 0, "Training dataset is empty."

test_dataset = DistilDataset(
    csv_files=[f"{test_data_path}/{t[0]}.csv" for t in distill_targets],
    npy_files=[f"{test_data_path}/{t[0]}.npy" for t in distill_targets],
    img_dir=test_data_path,
    transform=transforms.Compose(
    [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=test_img_mu, std=test_img_std),
    ]
),
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
assert len(test_dataset) > 0, "Testing dataset is empty."

# ---- Full Model w/ Heads

class FullModel(nn.Module):
    def __init__(self, num_tokens, token_dim, distill_targets):
        super(FullModel, self).__init__()
        self.backbone = Block(
            num_tokens=num_tokens,
            token_dim=token_dim,
        )
        self.heads = nn.ModuleDict()
        for name, (seq_len, hidden_dim) in distill_targets.items():
            head = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, seq_len * hidden_dim),
                nn.Reshape(seq_len, hidden_dim)
            )
            self.heads[name] = head

    def forward(self, x):
        x = self.backbone(x)
        return {name: head(x) for name, head in self.heads.items()}


# ----- Loss Function

def distillation_loss(outputs, targets):
    loss = 0
    for name, output in outputs.items():
        target = targets[name]
        seq_len, hidden_dim = output.shape[1], output.shape[2]
        target_seq_len, target_hidden_dim = target.shape[1], target.shape[2]
        
        if seq_len > target_seq_len:
            output = output[:, :target_seq_len, :]
        elif seq_len < target_seq_len:
            padding = torch.zeros(output.shape[0], target_seq_len - seq_len, hidden_dim).to(output.device)
            output = torch.cat([output, padding], dim=1)
        
        if hidden_dim != target_hidden_dim:
            raise ValueError(f"Hidden dimensions do not match for {name}: {hidden_dim} != {target_hidden_dim}")
        
        loss += nn.MSELoss()(output, target)
    
    return loss / len(outputs)

# ----- Optimizer

opt = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

# ---- Training Loop

print("Starting training...")
hist_loss_train = []
hist_loss_test = []
best_loss = np.inf
last_best_epoch = 0
for epoch in range(args.num_epochs):
    epoch_start_time = time.time()

    model.train()
    loss_train = 0
    progress_bar = tqdm(train_loader, desc=f"train.epoch.{epoch}")
    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)
        opt.zero_grad()
        outputs = model(images)
        loss = distillation_loss(outputs, targets)
        loss.backward()
        opt.step()
        loss_train += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    epoch_duration = time.time() - epoch_start_time
    print(f"\t duration {epoch_duration:0.2f} sec")

    # TRAIN LOSS
    loss_train /= len(train_loader)
    hist_loss_train.append(loss_train)
    print(f"\t loss/train: {loss_train}")
    writer.add_scalar("loss/train", loss_train, epoch)

    # EVAL LOSS
    with torch.no_grad():
        loss_test = 0
        progress_bar = tqdm(test_loader, desc="test")
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = distillation_loss(outputs, targets)
            loss_test += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    loss_test /= len(test_dataset)
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
    torch.save(model.state_dict(), model_save_path)

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
