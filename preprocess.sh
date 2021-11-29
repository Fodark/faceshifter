#!/bin/bash
# build docker image from Dockerfile
docker build -t dvl-faceshifter-dlib:devel ./preprocess
# run docker container from image
docker run -it --ipc host -v $PWD/preprocess:/workspace -v /datasets/faceshifter-datasets:/DATA -v /datasets/faceshifter-datasets-preprocessed:/RESULT --name dlib --tag dvl-faceshifter-dlib:devel "python preprocess.py --root /DATA --output_dir /RESULT"