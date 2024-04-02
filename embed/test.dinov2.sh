export DATA_PATH="/home/oop/dev/data"
# image dataste should be inside data path
export IMAGE_DIR="sdxl_imagenet_8/train"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
docker build -t embed.dinov2 -f Dockerfile.dinov2 .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="dinov2-small" \
    -e BATCH_SIZE=256 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.dinov2