# https://huggingface.co/openai/clip-vit-large-patch14
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import time
from transformers import CLIPProcessor, CLIPVisionModel
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='sdxl_imagenet_8')
parser.add_argument("--model", type=str, default='clip-vit-base-patch32')
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

models = [
    'clip-vit-base-patch16',
    'clip-vit-base-patch32',
    'clip-vit-large-patch14',
    'clip-vit-large-patch14-336',
]
assert args.model in models, f"Model {args.model} not found. Choose from {models}"
# https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel
model = CLIPVisionModel.from_pretrained(f"openai/{args.model}").to(device)
# https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPProcessor
processor = CLIPProcessor.from_pretrained(f"openai/{args.model}")
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

# Determine the sequence length based on the first image
with torch.no_grad():
    first_batch = next(iter(dataloader))
    first_outputs = model(**first_batch)
    sequence_length = first_outputs.last_hidden_state.shape[1]

# https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
hidden_size = model.config.hidden_size
embeddings = np.zeros((num_images, sequence_length, hidden_size), dtype=np.float32)
print(f"Embeddings will be {embeddings.shape} of {embeddings.dtype}")

for i, batch in enumerate(dataloader):
    start_time = time.time()
    print(f"Processing batch {i+1}/{len(dataloader)}...")
    with torch.no_grad():
        outputs = model(**batch)
    last_hidden_states = outputs.last_hidden_state
    start_index = i * args.batch_size
    end_index = start_index + len(last_hidden_states)
    embeddings[start_index:end_index] = last_hidden_states.cpu().numpy()
    print(f"\t... completed in {time.time()-start_time:.2f} seconds")

embeddings_path = os.path.join(image_dir_path, f'{args.model}.npy')
np.save(embeddings_path, embeddings)
print(f"Saved embeddings to {embeddings_path}")
df = pd.DataFrame(index=range(num_images), columns=['image_path'])
df['image_path'] = images
csv_path = os.path.join(image_dir_path, f'{args.model}.csv')
df.to_csv(csv_path, index=False)
print(f"Saved CSV to {csv_path}")