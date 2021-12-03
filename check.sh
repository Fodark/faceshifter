#!/bin/bash
# build docker image from Dockerfile
docker build -t dvl-faceshifter-dlib-lzanella:devel ./preprocess
# run docker container from image
docker run -it --ipc host -v $PWD:/workspace -v ~/datasets/faceshifter-datasets:/DATA dvl-faceshifter-dlib-lzanella:devel ./run_checking.sh