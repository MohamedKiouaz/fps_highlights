from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.metrics import accuracy
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet18
from loguru import logger as log
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from ml import create_model


def plot_roc_curve(y_test, y, path):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y)
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path)

def plot_n_images(images, pred, true, path):
    # Display the most uncertain images
    n_images = len(images)
    n_rows = int(np.sqrt(n_images))
    n_cols = int(np.sqrt(n_images))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5*n_rows, 2.5*n_cols))

    for i, ax in enumerate(axs.flat):
        if i > len(axs.flat):
            break
        ax.imshow(images[i])
        # set title to TP, FP, TN, FN
        title = ''
        if pred[i] > 0.5 and true[i] > 0.5:
            title = 'TP'
        elif pred[i] > 0.5 and true[i] < 0.5:
            title = 'FP'
        elif pred[i] < 0.5 and true[i] < 0.5:
            title = 'TN'
        elif pred[i] < 0.5 and true[i] > 0.5:
            title = 'FN'
            
        ax.set_title(f'{title} Pred:{pred[i]:.2f}')
        ax.axis('off')

    fig.tight_layout()
    
    fig.savefig(path)
    log.info(f'Saved {path}')

def plot_n_images_by_idxs(images, pred, true, path, idxs):
    """
    Plots n images by indexes
    """

    sorted_images = [images[i] for i in idxs]
    sorted_predictions = [pred[i] for i in idxs]
    sorted_y_test = [true[i] for i in idxs]

    plot_n_images(sorted_images, sorted_predictions, sorted_y_test, path)

if __name__ == '__main__':
    model, data = create_model()

    save_path = Path('graphs')
    save_path.mkdir(exist_ok=True)

    log.info(f'Predicting {len(data.valid_ds)} images...')
    y_test = np.array([img[1].numpy() for img in data.valid_ds])
    y = np.array([model.predict(img[0])[2][1].numpy() for img in tqdm(data.valid_ds)])
    imgs = [np.asarray(img[0]) for img in data.valid_ds]

    sorted_idxs = np.argsort(y_test)
    sorted_idxs = [e for i,e in enumerate(sorted_idxs) if i % 10 == 0]
    sorted_idxs = np.concatenate([sorted_idxs[:50], sorted_idxs[-50:]])
    plot_n_images_by_idxs(imgs, y, y_test, save_path / 'random_images.png', sorted_idxs)

    # Plot ROC curve
    log.info('Plotting ROC curve...')
    plot_roc_curve(y_test, y, save_path / 'roc.png')

    # Plot most uncertain images
    log.info('Plotting most uncertain images...')
    certainties =  abs(0.5 - y)
    sorted_idxs = np.argsort(certainties)

    plot_n_images_by_idxs(imgs, y, y_test, save_path / 'most_uncertain.png', sorted_idxs[:25])

    sorted_y_test = [y_test[i] for i in sorted_idxs]

    # indexes of true values
    false_idxs = np.where(np.array(sorted_y_test) == 0)[0]

    plot_n_images_by_idxs(imgs, y, y_test, save_path / 'most_uncertain_false.png', false_idxs[:25])

    # indexes of true values
    true_idxs = np.where(np.array(sorted_y_test) == 1)[0]

    plot_n_images_by_idxs(imgs, y, y_test, save_path / 'most_uncertain_true.png', true_idxs[:25])
