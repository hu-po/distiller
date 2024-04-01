import pandas as pd
import numpy as np

# datasets = [
#     "/home/oop/dev/data/sdxl_imagenet_8/clip-vit-large-patch14-336.csv",
#     "/home/oop/dev/data/sdxl_imagenet_8/dinov2-giant.csv",
#     "/home/oop/dev/data/sdxl_imagenet_8/siglip-large-patch16-384.csv",
# ]

datasets = [
    "/home/oop/dev/data/sdxl_imagenet_8/clip-vit-base-patch16.csv",
    "/home/oop/dev/data/sdxl_imagenet_8/dinov2-small.csv",
    "/home/oop/dev/data/sdxl_imagenet_8/siglip-base-patch16-224.csv",
]

for dataset in datasets:
    df = pd.read_csv(dataset)
    print(f"Dataset {dataset} has {df.shape[0]} rows")