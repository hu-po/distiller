export DATA_PATH="/home/oop/dev/data"
# image dataste should be inside data path
export IMAGE_DIR="sdxl_imagenet_8/test"
export HF_CACHE_PATH="/home/oop/.cache/huggingface"
# pixel statistics of image directory
pixel_stats=$(python pixelstats.py --dir ${DATA_PATH}/${IMAGE_DIR})
img_mu=$(echo "$pixel_stats" | sed -n '1p')
img_std=$(echo "$pixel_stats" | sed -n '2p')
# DinoV2
docker build -t embed.dinov2 -f Dockerfile.dinov2 .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="dinov2-small" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    -e IMG_MU=${img_mu} \
    -e IMG_STD=${img_std} \
    embed.dinov2
# CLIP
docker build -t embed.clip -f Dockerfile.clip .
docker run \
    -it \
    --rm \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e MODEL="clip-vit-base-patch16" \
    -e BATCH_SIZE=128 \
    -e IMAGE_DIR=${IMAGE_DIR} \
    -e IMG_MU=${img_mu} \
    -e IMG_STD=${img_std} \
    embed.clip
# SigLIP
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
    -e IMG_MU=${img_mu} \
    -e IMG_STD=${img_std} \
    embed.siglip