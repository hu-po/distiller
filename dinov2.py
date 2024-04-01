import argparse
import os
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/data/sdxl_imagenet_8')
parser.add_argument("--model", type=str, default='facebook/dinov2-small')
args = parser.parse_args()

# https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3
models = [
    'facebook/dinov2-small',
    'facebook/dinov2-base',
    'facebook/dinov2-large',
    'facebook/dinov2-giant',
]
assert args.model in models, f"Model {args.model} not found. Choose from {models}"
processor = AutoImageProcessor.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model)
print(f"Loaded model {args.model}")
df = pd.DataFrame(columns=['image_path', 'embedding'])
assert os.path.exists(args.image_dir), f"Directory {args.image_dir} not found"
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
images = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
print(f"Found {len(images)} images in directory {args.image_dir}")
for image_path in images:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embedding = last_hidden_states.squeeze().detach().numpy()
    df = df.append({'image_path': image_path, 'embedding': embedding}, ignore_index=True)
csv_path = os.path.join(args.image_dir, 'dinov2.csv')
df.to_csv(csv_path, index=False)