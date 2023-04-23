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

    log.info("Predicting.")
    # predict all images and the uncertain ones to a separate folder
    # the reasoning behind this is that maybe those images are not in the right class
    # errors in the labeling process could have happened
    # this would explain why the model has troubles with them
    # this way we can manually check them and fix if needed
    pred_values, _ = learn.get_preds(0)
    confidences = abs(pred_values.numpy()[:, 0] - 0.5)
    log.info(f'{confidences.mean()*100:.2f}% of the train set is uncertain.')
    for i, e in tqdm(enumerate(confidences)):
        if e < 0.45:
            data.train_ds.items[i].rename(Path('low_confidence') / data.train_ds.items[i].parent.name / data.train_ds.items[i].name)

    pred_values, _ = learn.get_preds(1)
    confidences = abs(pred_values.numpy()[:, 0] - 0.5)
    log.info(f'{confidences.mean()*100:.2f}% of the validation set is uncertain.')
    for i, e in tqdm(enumerate(confidences)):
        if e < 0.45:
            data.valid_ds.items[i].rename(Path('low_confidence') / data.valid_ds.items[i].parent.name / data.valid_ds.items[i].name)