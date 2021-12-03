import os
import random

base_dir = '/raid/home/dvl/datasets/faceshifter-datasets-preprocessed'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

file_list = os.listdir(base_dir)
random.shuffle(file_list)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# grab indices for train and val
train_indices = file_list[:int(0.8 * len(file_list))]
val_indices = file_list[int(0.8 * len(file_list)):]

# move files to train and val
for file in train_indices:
    os.rename(os.path.join(base_dir, file), os.path.join(train_dir, file))

for file in val_indices:
    os.rename(os.path.join(base_dir, file), os.path.join(val_dir, file))