from pathlib import Path

from fastai.metrics import RocAuc, accuracy
from fastai.vision.augment import RandomErasing, RandomResizedCrop, Resize
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet18
from loguru import logger as log

from fps_functions import create_folders


def create_model(train=False):
    path = Path('inputs')

    log.info("Loading data.")
    data = ImageDataLoaders.from_folder(path, train='train', valid='valid') 
    
    data.add_tfms(Resize((100, 100), method='squish'), 'after_item')
    
    if train:
        data.add_tfms(RandomResizedCrop((100, 100), min_scale=0.90), 'after_item')
        #data.add_tfms(RandomErasing(), 'after_item')

    log.info("Creating Model.")
    model = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy,RocAuc()])

    if Path('inputs/models/resnet18.pth').exists():
        log.info("Loading weights.")
        model = model.load('resnet18')

    log.info("Done.")
    return model, data

if __name__ == '__main__':
    epochs = 1

    create_folders()

    learn, data = create_model(train=True)
    
    log.info(f'Training set: {len(data.train_ds)}')
    log.info(f'Validation set: {len(data.valid_ds)}')
    
    log.info("Creating model summary file.")
    with open('inputs/models/summary.txt', 'w+') as f:
        f.write(learn.summary())

    for _ in range(epochs):
        log.info("Training.")
        # dataset has 180k images, so 1 epoch is enough
        learn.fit(1)
        log.info(f'Epoch {learn.epoch} done.')
      
        log.info("Saving model.")
        learn.save('resnet18')