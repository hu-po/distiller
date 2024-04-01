import argparse
import os
import pandas as pd
from PIL import Image
import time
from transformers import AutoImageProcessor, AutoModel
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='sdxl_imagenet_8')
parser.add_argument("--model", type=str, default='dinov2-base')
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

models = [
    'dinov2-small',
    'dinov2-base',
    'dinov2-large',
    'dinov2-giant',
]
assert args.model in models, f"Model {args.model} not found. Choose from {models}"
processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model}")
model = AutoModel.from_pretrained(f"facebook/{args.model}").to(device)
print(f"Loaded model {args.model}")
image_dir_path = os.path.join('/data', args.image_dir)
assert os.path.exists(image_dir_path), f"Directory {image_dir_path} not found"
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
images = []
for root, dirs, files in os.walk(image_dir_path):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_path = os.path.join(root, file)
            images.append(image_path)
num_images = len(images)
print(f"Found {num_images} images in {image_dir_path}")

def load_image(image_paths):
    return [Image.open(path) for path in image_paths]

def collate_fn(examples):
    image_paths = [example["image_path"] for example in examples]
    images = load_image(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

dataset = Dataset.from_dict({"image_path": images})
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

df = pd.DataFrame(index=range(num_images), columns=['image_path', 'embedding'])
for i, batch in enumerate(dataloader):
    start_time = time.time()
    print(f"Processing batch {i+1}/{len(dataloader)}...")
    with torch.no_grad():
        outputs = model(**batch)
    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states.squeeze().cpu().numpy()
    start_index = i * args.batch_size
    end_index = min((i + 1) * args.batch_size, num_images)
    df.loc[start_index:end_index-1, 'image_path'] = images[start_index:end_index]
    df.loc[start_index:end_index-1, 'embedding'] = embeddings[:end_index-start_index].tolist()
    print(f"\t... completed in {time.time()-start_time:.2f} seconds")
print("Saving embeddings to CSV...")
csv_path = os.path.join(image_dir_path, f'{args.model}.csv')
df.to_csv(csv_path, index=False)