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
    "/home/oop/dev/data/sdxl_imagenet_8/siglip-base-patch16-224",
]

# Each CSV contains two columns: image_path, embedding
# embeddings = last_hidden_states.squeeze().cpu().numpy()
# df = pd.DataFrame(index=range(num_images), columns=['image_path', 'embedding'])

# Open all the CSVs, print out the size of the embeddings
for dataset in datasets:
    df = pd.read_csv(dataset)
    print(f"Dataset {dataset} has {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Embedding size: {df['embedding'][0].shape}")
