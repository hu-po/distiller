export DATA_PATH="/home/oop/dev/data"
# image dataste should be inside data path
export IMAGE_DIR="sdxl_imagenet_8"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
docker build -t embed.siglip -f Dockerfile.siglip .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="siglip-base-patch16-224" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.siglip