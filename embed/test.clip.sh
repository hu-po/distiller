export DATA_PATH="/home/oop/dev/data"
# image dataste should be inside data path
export IMAGE_DIR="sdxl_imagenet_8"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
docker build -t embed.clip -f Dockerfile.clip .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="clip-vit-base-patch16" \
    -e BATCH_SIZE=256 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.clip