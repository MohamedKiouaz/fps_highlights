from pathlib import Path
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, RocAucBinary
from fastai.vision.models import resnet18
from fastai.interpret import Interpretation
from loguru import logger as log

if __name__ == '__main__':
    path = Path('inputs')
    epochs = 30

    data = ImageDataLoaders.from_folder(path, train='train', valid='valid')
    log.info(f'training set: {len(data.train_ds)}')
    log.info(f'validation set: {len(data.valid_ds)}')

    log.info("Creating Model.")
    input_shape = data.one_batch()[0].shape[1:]
    learn = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy,RocAucBinary()])
    
    log.info("Loading weights.")
    learn.load('resnet18')

    log.info("Creating model summary file.")
    with open('inputs/models/summary.txt', 'w') as f:
        f.write(learn.summary())

    log.info("Showing results.")
    #interpretation = Interpretation.from_learner(learn)
    #interpretation.show_results([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    log.info("Validating.")
    log.info(learn.validate())

    log.info("Printing confusion matrix.")
    log.info(learn.show_results())

    for _ in range(epochs//5):
        log.info("Training.")
        learn.fit(5)

        learn.save('resnet18')
        log.info(f'Epoch {learn.epoch} done.')