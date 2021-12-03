from PIL import Image
import os

train_path = os.path.join('/DATA', 'train')
val_path = os.path.join('/DATA', 'val')

# verify with PIL integrity of every file in train_path
for filename in os.listdir(train_path):
    img = Image.open(os.path.join(train_path, filename))
    try:
        img.verify()
    except Exception:
        print('Corrupted image in train: ' + filename)

# also for val
for filename in os.listdir(val_path):
    img = Image.open(os.path.join(val_path, filename))
    try:
        img.verify()
    except Exception:
        print('Corrupted image in val: ' + filename)