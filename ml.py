from pathlib import Path

from fastai.metrics import RocAucBinary, accuracy
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet18
from fastai.vision.augment import Resize
from loguru import logger as log

from apex_functions import create_folders

from tqdm import tqdm

def create_model():
    path = Path('inputs')

    log.info("Loading data.")
    data = ImageDataLoaders.from_folder(path, train='train', valid='valid') 
    
    # resize to 100x100
    data.add_tfms(Resize((100, 100), method='squish'), 'after_item')

    log.info("Creating Model.")
    model = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy,RocAucBinary()])

    if Path('inputs/models/resnet18.pth').exists():
        log.info("Loading weights.")
        model = model.load('resnet18')

    log.info("Done.")
    return model, data

if __name__ == '__main__':
    epochs = 30

    create_folders()

    learn, data = create_model()
    
    log.info(f'Training set: {len(data.train_ds)}')
    log.info(f'Validation set: {len(data.valid_ds)}')
    
    log.info("Creating model summary file.")
    with open('inputs/models/summary.txt', 'w+') as f:
        f.write(learn.summary())

    #log.info("Showing results.")
    #interpretation = Interpretation.from_learner(learn)
    #interpretation.show_results([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    #log.info("Validating.")
    #log.info(learn.validate())

    log.info("Printing confusion matrix.")
    log.info(learn.show_results())

    for _ in range(epochs):
        log.info("Training.")
        learn.fit(1)
        log.info(f'Epoch {learn.epoch} done.')
      
        log.info("Saving model.")
        learn.save('resnet18')