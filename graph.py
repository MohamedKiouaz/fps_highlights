from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.metrics import accuracy
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet18

if __name__ == '__main__':
    epochs = 30

    path = Path('inputs')

    data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.05)

    learn = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy])
    learn.load("resnet18")
    
    predictions = []
    uncertainties = []
    for img in data.train_ds:
        prediction = learn.predict(img[0])[2].numpy()
        predictions.append(prediction)
        uncertainty = np.max(prediction)
        uncertainties.append(uncertainty)
    
    sorted_idxs = np.argsort(uncertainties)
    sorted_images = [data.train_ds[i] for i in sorted_idxs]
    sorted_predictions = [predictions[i] for i in sorted_idxs]

    # Display the most uncertain images
    n_images = 25
    fig, axs = plt.subplots(5, 5)
    for i, ax in enumerate(axs.flat):
        # how to show a pilimage
        ax.imshow(np.asarray(sorted_images[i][0]))
        # add title
        ax.set_title(f'{sorted_predictions[i][0]:.2f} {sorted_predictions[i][1]:.2f}')
        ax.axis('off')

    # tight layout
    plt.tight_layout()
    plt.show()

