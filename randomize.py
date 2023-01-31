import glob
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    print('Randomizing...')

    images = glob.glob('inputs/**/*.png', recursive=True)

    ratio = 0.05

    train = [k for k in images if 'train' in k]
    valid = [k for k in images if 'valid' in k]

    print(f'Found {len(train)} train images.')
    print(f'Found {len(valid)} valid images.')

    train = np.random.choice(train, int(len(train) * ratio), replace=False)
    valid = np.random.choice(valid, int(len(valid) * (1 - ratio)), replace=False)

    # This part is not 100% scientifically valid
    # (images from the same video could be used in both train and valid)
    # but it should work well enough for our purposes.

    for path in tqdm(train):
        # move to valid
        new_path = path.replace('train', 'valid')
        shutil.move(path, new_path)

    for path in tqdm(valid):
        # move to train
        new_path = path.replace('valid', 'train')
        shutil.move(path, new_path)              

    print('Done.')

    shutil.rmtree('inputs/models', ignore_errors=True)

    print('Removed models.')

    print('Verifying image integrity...')
    # The following code tries to open all images and removes them if they are corrupt.
    # This is needed because the handling of corrupt images is not very well
    # handled in fastai.
    # The code only removes the first 10 corrupt images.
    # This is a safety measure to prevent any data loss.
    # We do not expect any corrupt images in the first place.
    i = 0
    images = glob.glob('inputs/**/*.png', recursive=True)
    for image in tqdm(images):
        try:
            img = Image.open(image)
            img.verify()
        except Exception:
            print(f'Failed to open {image}')
            if i < 10:
                #remove
                os.remove(image)
                i += 1
                print(f'Removed {image}')