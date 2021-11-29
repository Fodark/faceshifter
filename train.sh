#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1
docker build -t dvl-faceshifter-lzanella:devel .
docker run -it --ipc host --gpus all -v $PWD:/workspace -v /datasets/faceshifter-datasets-preprocessed:/DATA dvl-faceshifter-lzanella:devel "python aei_trainer.py -c config/train.yaml -n buona_la_prima"