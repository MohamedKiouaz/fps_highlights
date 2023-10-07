from pathlib import Path

from loguru import logger as log
from tqdm import tqdm

from fps_functions import create_folders
from ml import create_model

if __name__ == '__main__':
    epochs = 5

    create_folders()

    learn, data = create_model(train=True)
    
    log.info(f'Training set: {len(data.train_ds)}')
    log.info(f'Validation set: {len(data.valid_ds)}')

    # predict all images and move the most certain ones to a separate folder
    # the reasoning behind this is that maybe those images are not required in the class for an optimal training
    # this way we can manually check them and remove if needed
 
    prune_classes = ['false', 'rank']
    
    log.info("Predicting.") 
    pred_values, targs = learn.get_preds(0)
    confidences = abs(pred_values.numpy().max(axis=1) - 0.5)
    threshold = 0.48
    log.info(f'{sum(confidences > threshold)/confidences.shape[0]*100:.2f}% of the train set is too certain and is maybe not providing enough value.')
    for i, e in tqdm(enumerate(confidences)):
        if e > threshold:
            data.train_ds.items[i].rename(Path('inputs/high_confidence') / data.train_ds.items[i].parent.name / data.train_ds.items[i].name)