from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.metrics import accuracy
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet18
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


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

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for i, ax in enumerate(axs.flat):
        if i > len(axs.flat):
            break
        ax.imshow(images[i])
        ax.set_title(f'Pred:{pred[i]:.2f} GT:{true[i]:.2f}')
        ax.axis('off')

    fig.tight_layout()
    
    fig.savefig(path)
    print(f'Saved {path}')


if __name__ == '__main__':
    path = Path('inputs')
    data = ImageDataLoaders.from_folder(path, train='train', valid='valid')

    learn = vision_learner(data, resnet18, bn_final=True, model_dir="models", metrics=[accuracy])
    learn.load("resnet18")

    save_path = Path('graphs')
    save_path.mkdir(exist_ok=True)
    
    print(f'Predicting {len(data.valid_ds)} images...')
    y_test = np.array([img[1].numpy() for img in data.valid_ds])
    y = np.array([learn.predict(img[0])[2][1].numpy() for img in data.valid_ds])
    imgs = [np.asarray(img[0]) for img in data.valid_ds]

    print('Plotting ROC curve...')    
    # Plot ROC curve
    plot_roc_curve(y_test, y, save_path / 'roc.png')

    print('Plotting most uncertain images...')
    # Plot most uncertain images
    certainties =  abs(0.5 - y)
    sorted_idxs = np.argsort(certainties)[:25]
    sorted_images = [imgs[i] for i in sorted_idxs]
    sorted_predictions = [y[i] for i in sorted_idxs]
    sorted_y_test = [y_test[i] for i in sorted_idxs]

    plot_n_images(sorted_images, sorted_predictions, sorted_y_test, save_path / 'most_uncertain.png')

    sorted_idxs = np.argsort(certainties)
    sorted_images = [imgs[i] for i in sorted_idxs]
    sorted_predictions = [y[i] for i in sorted_idxs]
    sorted_y_test = [y_test[i] for i in sorted_idxs]

    # indexes of true values
    false_idxs = np.where(np.array(sorted_y_test) == 0)[0]
    false_images = [sorted_images[i] for i in false_idxs][:25]
    false_predictions = [y[i] for i in false_idxs][:25]
    false_y_test = [sorted_y_test[i] for i in false_idxs][:25]

    plot_n_images(false_images, false_predictions, false_y_test, save_path / 'most_uncertain_false.png')

    # indexes of true values
    true_idxs = np.where(np.array(sorted_y_test) == 1)[0]
    true_images = [sorted_images[i] for i in true_idxs][:25]
    true_predictions = [y[i] for i in true_idxs][:25]
    true_y_test = [sorted_y_test[i] for i in true_idxs][:25]

    plot_n_images(true_images, true_predictions, true_y_test, save_path / 'most_uncertain_true.png')
