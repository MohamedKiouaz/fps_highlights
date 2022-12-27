import fastai
import fastai.vision as vision
from pathlib import Path
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy


if __name__ == '__main__':
    epochs = 30

    path = Path('inputs')

    data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.05)
    print(f'training set: {data.train_ds}')
    print(f'validation set: {len(data.valid_ds)}')

    input_shape = data.one_batch()[0].shape[1:]
    learn = vision_learner(data, vision.models.resnet50, bn_final=True, model_dir="models", metrics=[accuracy])
    

    learn = vision_learner(data, vision.models.resnet50, bn_final=True, model_dir="models")

    try:
        learn.load('model')
    except Exception:
        print("Cannot load model, starting from scratch.")

    for _ in range(epochs):
        learn.fit(1)

        learn.save('model')