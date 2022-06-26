docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 \
    -p 8888:8888 -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${PWD}:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:22.05-py3 bash
