from pathlib import Path
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, RocAucBinary
from fastai.vision.models import resnet18
from fastai.interpret import Interpretation


if __name__ == '__main__':
    epochs = 30

    path = Path('inputs')

    data = ImageDataLoaders.from_folder(path, train='train', valid='valid')
    print(f'training set: {len(data.train_ds)}')
    print(f'validation set: {len(data.valid_ds)}')

    print("Creating Model.")
    input_shape = data.one_batch()[0].shape[1:]
    learn = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy,RocAucBinary()])
    
    print("Loading weights.")
    learn.load('resnet18')

    print("Printing summary.")
    with open('inputs/models/summary.txt', 'w') as f:
        f.write(learn.summary())

    print("Printing results.")
    interpretation = Interpretation.from_learner(learn)
    interpretation.show_results([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print("Validating.")
    learn.validate()
    print(learn.validate())

    print("Printing confusion matrix.")
    learn.show_results()

    for _ in range(epochs//5):
        print("Training.")
        learn.fit(5)

        learn.save('resnet18')
        print(f'Epoch {learn.epoch} done.')