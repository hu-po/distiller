import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np

# When running in the docker container the model will be in /src/model.py
if os.path.exists("/src/model.py"):
    from model import Block
else:
    # for local testing use hardcoded path
    from evolve.models.pytorch.mlp import Block

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test.pytorch")
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

parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--learning_rate_gamma", type=float, default=0.9)

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

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        # TODO: Normalize to generated dataset?
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = ImageFolder(root=args.train_data_dir, transform=preprocess)
test_dataset = ImageFolder(root=args.test_data_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
assert len(train_dataset) > 0, "Training dataset is empty."
assert len(test_dataset) > 0, "Testing dataset is empty."

num_classes = len(train_dataset.classes)
model = Block(num_classes=num_classes).to(device)
# Verify output shape
for param in model.parameters():
    assert param.sum() != 0, "Model parameter(s) not initialized properly."
assert model(torch.randn(1, 3, 224, 224).to(device)).shape == torch.Size([1, num_classes])
model_size = sum(p.numel() for p in model.parameters())
assert model_size < args.max_model_size, f"Model size {model_size} exceeds limit {args.max_model_size}"
print(f"Model size: {model_size / 1e6}M")
hparams["model_size"] = sum(p.numel() for p in model.parameters()),

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_rate_gamma)

best_loss = float("inf")
last_best_epoch = 0
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"train.epoch {epoch}")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / len(progress_bar)})
    print(f"epoch {epoch}, loss: {running_loss / len(train_loader)}")
    writer.add_scalar("loss/train", running_loss / len(train_loader), epoch)
    if running_loss < best_loss:
        best_loss = running_loss
        last_best_epoch = epochtensorboardX
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="test")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_accuracy += (predicted == labels).sum().item()
            progress_bar.set_postfix({"acc": test_accuracy / len(progress_bar)})
    test_accuracy /= len(test_dataset)
    print(f"acc/test: {test_accuracy}")
    writer.add_scalar("acc.test", test_accuracy, epoch)
    if epoch - last_best_epoch > args.early_stop:
        print(f"early stopping at epoch {epoch}")
        break
if args.save_ckpt:
    print(f"Saving model to {args.ckpt_dir}/{args.run_name}.e{epoch}.pth")
    torch.save(model.state_dict(), f"{args.ckpt_dir}/{args.run_name}.e{epoch}.pth")
scores = {
    "test_accuracy": test_accuracy,
}
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