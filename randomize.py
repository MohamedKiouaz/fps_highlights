import glob
import os
import shutil

import numpy as np
from tqdm import tqdm

from loguru import logger as log
from fastai.vision.utils import verify_images

if __name__ == '__main__':
    log.info('Randomizing...')

    images = glob.glob('inputs/**/*.png', recursive=True)

    ratio = 0.1

    train = [k for k in images if 'train' in k]
    valid = [k for k in images if 'valid' in k]

    log.info(f'Found {len(train)} train images.')
    log.info(f'Found {len(valid)} valid images.')

    train = np.random.choice(train, int(len(train) * ratio), replace=False)
    valid = np.random.choice(valid, int(len(valid) * (1 - ratio)), replace=False)

    # This part is not 100% scientifically valid
    # (images from the same video could be used in both train and valid)
    # but it should work well enough for our purposes.

    log.info(f'Moving {len(train)} randomly selected train images.')
    for path in tqdm(train):
        new_path = path.replace('train', 'valid')
        shutil.move(path, new_path)

    log.info(f'Moving {len(valid)} randomly selected valid images.')
    for path in tqdm(valid):
        new_path = path.replace('valid', 'train')
        shutil.move(path, new_path)              

    log.info('Done.')

    shutil.rmtree('inputs/models', ignore_errors=True)

    log.info('Removed models.')

    log.info('Verifying image integrity...')
    # The following code tries to open all images and removes them if they are corrupt.
    # This is needed because the handling of corrupt images is not very well
    # done in fastai.

    images = glob.glob('inputs/**/*.png', recursive=True)

    unreadable_images = verify_images(images)

    if (len(unreadable_images) > 0):
        log.warning(f'Found {len(unreadable_images)} unreadable images. Removing them...')

        if (len(unreadable_images) > 100):
            log.error('More than 100 unreadable images. Aborting.')
            for img in unreadable_images:
                log.error(f'{img} is unreadable.')
        else:
            for img in unreadable_images:
                log.info(f'{img} is unreadable. Removing it.')
                os.remove(img)