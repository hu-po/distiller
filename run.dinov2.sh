export DATA_PATH="/home/oop/dev/data"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
docker build -t embed.dinov2 -f Dockerfile.dinov2 .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    embed.dinov2