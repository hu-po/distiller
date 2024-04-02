export DATA_PATH="/home/oop/dev/data"
export CKPT_PATH="/home/oop/dev/data/test_model/ckpt"
export LOGS_PATH="/home/oop/dev/data/test_model/logs"
export MODEL_PATH="/home/oop/dev/distiller/evolve/models/pytorch/mlp.py"
# image dataset should be inside data path
export TRAIN_IMAGE_DIR="sdxl_imagenet_8/train"
export TEST_IMAGE_DIR="sdxl_imagenet_8/test"
docker build \
     -t "evolve.pytorch" \
     -f Dockerfile.pytorch .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    -v ${MODEL_PATH}:/src/model.py \
    evolve.pytorch \
    python /src/traineval.pytorch.py \
    --run_name=test \
    --round=0 \
    --train_data_dir=$TRAIN_IMAGE_DIR \
    --test_data_dir=$TEST_IMAGE_DIR