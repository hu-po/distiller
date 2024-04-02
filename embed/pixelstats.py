import argparse
import os
import sys
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=".")
args = parser.parse_args()

directory = args.dir
assert os.path.isdir(directory), f"Error: {directory} is not a valid directory."
print(f"Calculating pixel statistics for images in {directory}...")

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
image_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_files.append(os.path.join(root, file))

if not image_files:
    print(f"No image files found in {directory}.")
    sys.exit(0)

pixel_values = []

for image_file in image_files:
    with Image.open(image_file) as img:
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
        pixel_values.append(img_array)

pixel_values = np.concatenate(pixel_values)
mean = np.mean(pixel_values, axis=(0, 1))
std = np.std(pixel_values, axis=(0, 1))

# Format mean and std as comma-separated strings
img_mu = ",".join([f"{val:.3f}" for val in mean])
img_std = ",".join([f"{val:.3f}" for val in std])

print(f"--img_mu {img_mu}")
print(f"--img_std {img_std}")