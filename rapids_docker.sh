docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 \
    -p 8888:8888 \
    -v ${PWD}:/workspace \
    -w /workspace \
    rapidsai/rapidsai-core:22.04-cuda11.4-base-ubuntu20.04-py3.9
