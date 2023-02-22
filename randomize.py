import glob
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from loguru import logger as log

def verify_image(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception:
        log.warning(f'Failed to open {img_path}')
        os.remove(img_path)
        log.warning(f'Removed {img_path}')

if __name__ == '__main__':
    log.info('Randomizing...')

    images = glob.glob('inputs/**/*.png', recursive=True)

    ratio = 0.05

    train = [k for k in images if 'train' in k]
    valid = [k for k in images if 'valid' in k]

    log.info(f'Found {len(train)} train images.')
    log.info(f'Found {len(valid)} valid images.')

    train = np.random.choice(train, int(len(train) * ratio), replace=False)
    valid = np.random.choice(valid, int(len(valid) * (1 - ratio)), replace=False)

    # This part is not 100% scientifically valid
    # (images from the same video could be used in both train and valid)
    # but it should work well enough for our purposes.

    for path in tqdm(train):
        new_path = path.replace('train', 'valid')
        shutil.move(path, new_path)

    for path in tqdm(valid):
        new_path = path.replace('valid', 'train')
        shutil.move(path, new_path)              

    log.info('Done.')

    shutil.rmtree('inputs/models', ignore_errors=True)

    log.info('Removed models.')

    log.info('Verifying image integrity...')
    # The following code tries to open all images and removes them if they are corrupt.
    # This is needed because the handling of corrupt images is not very well
    # handled in fastai.
    
    images = glob.glob('inputs/**/*.png', recursive=True)

    with ProcessPoolExecutor() as executor:
        executor.map(verify_image, tqdm(images))