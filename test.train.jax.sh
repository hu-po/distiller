export DATA_PATH="/home/oop/dev/data"
export CKPT_PATH="/home/oop/dev/data/test.train.jax/ckpt"
export LOGS_PATH="/home/oop/dev/data/test.train.jax/logs"
export MODEL_PATH="/home/oop/dev/distiller/models/jax/cnn.py"
# image dataset should be inside data path
export TRAIN_IMAGE_DIR="sdxl_imagenet_8/train"
export TEST_IMAGE_DIR="sdxl_imagenet_8/test"
docker build \
     -t "evolve.jax" \
     -f Dockerfile.jax .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    -v ${MODEL_PATH}:/src/model.py \
    evolve.jax \
    python /src/train.jax.py \
    --run_name=test \
    --round=0 \
    --train_data_dir=$TRAIN_IMAGE_DIR \
    --test_data_dir=$TEST_IMAGE_DIR \
    --num_epochs=8 \
    --batch_size=8