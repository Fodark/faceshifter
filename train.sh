#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1
docker build -t dvl-faceshifter-lzanella:devel .
docker run -it --ipc host --gpus all -v $PWD:/workspace -v ~/datasets/faceshifter-datasets-preprocessed:/DATA dvl-faceshifter-lzanella:devel ./run_training.sh