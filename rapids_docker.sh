docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 \
    -p 8887:8887 \
    -v ${PWD}:/workspace \
    -w /workspace \
    rapidsai/rapidsai-core:22.06-cuda11.5-base-ubuntu20.04-py3.8
