export DATA_PATH="/home/oop/dev/data/test_data"
export CKPT_PATH="/home/oop/dev/data/test_model/ckpt"
export LOGS_PATH="/home/oop/dev/data/test_model/logs"
export MODEL_PATH="/home/oop/dev/distiller/evolve/models/jax/mlp.py"
docker build \
     -t "evolve.jax" \
     -f Dockerfile.jax .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${MODEL_PATH}:/src/model.py \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    -e RUN_NAME=test \
    -e ROUND=0 \
    evolve.jax