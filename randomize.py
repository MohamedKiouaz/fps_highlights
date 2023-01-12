import glob
import shutil
import numpy as np
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