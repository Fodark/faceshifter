import os
import random

base_dir = '/raid/home/dvl/datasets/faceshifter-datasets-preprocessed'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# move 99% of images from val to dir
for root, dirs, files in os.walk(val_dir):
    for file in files:
        if random.random() < 0.01:
            os.rename(os.path.join(root, file), os.path.join(train_dir, file))

