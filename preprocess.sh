#!/bin/bash
# build docker image from Dockerfile
docker build -t dvl-faceshifter-dlib-lzanella:devel ./preprocess
# run docker container from image
docker run -it --ipc host -v $PWD:/workspace -v /datasets/faceshifter-datasets:/DATA -v /datasets/faceshifter-datasets-preprocessed:/RESULT dvl-faceshifter-dlib-lzanella:devel ./run_preprocess.sh