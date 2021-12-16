#!/bin/bash
CUDA_VISIBLE_DEVICES=0
docker build -t dvl-faceshifter-lzanella:devel .
docker run -it --ipc host --gpus all -v $PWD:/workspace -v /media/hdd_data/dvl1/:/DATA dvl-faceshifter-lzanella:devel ./run_celeba_test.sh