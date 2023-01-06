import fastai
import fastai.vision as vision
from pathlib import Path
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, RocAucBinary
from fastai.vision.models import resnet18


if __name__ == '__main__':
    epochs = 30

    path = Path('inputs')

    data = ImageDataLoaders.from_folder(path, train='train', valid='valid')
    print(f'training set: {len(data.train_ds)}')
    print(f'validation set: {len(data.valid_ds)}')

    input_shape = data.one_batch()[0].shape[1:]
    learn = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy,RocAucBinary()])
    
    learn.load('resnet18')

    with open('inputs/models/summary.txt', 'w') as f:
        f.write(learn.summary())

    learn.validate()
    print(learn.validate())

    learn.show_results()

    for _ in range(epochs//5):
        learn.fit(5)

        learn.save('resnet18')
        print(f'Epoch {learn.epoch} done.')