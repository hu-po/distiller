export DATA_PATH="/home/oop/dev/data"
# image dataste should be inside data path
export IMAGE_DIR="sdxl_imagenet_8/train"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
# pixel statistics of directory
python pixelstats.py --dir ${DATA_PATH}/${IMAGE_DIR}
# DinoV2
docker build -t embed.dinov2 -f Dockerfile.dinov2 .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="dinov2-giant" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.dinov2
# CLIP
docker build -t embed.clip -f Dockerfile.clip .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="clip-vit-large-patch14-336" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.clip
# SigLIP
docker build -t embed.siglip -f Dockerfile.siglip .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="siglip-large-patch16-384" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    embed.siglip