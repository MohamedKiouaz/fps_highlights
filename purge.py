# i have 2 folders with a lot of png images
# i want to purge the images that are too similar to each other
# i want to keep the images that are different from each other

import itertools
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


def get_image_list(path):
    return [file for file in os.listdir(path) if file.endswith(".png")]

def get_image_array(path, image_list):
    return np.array(
        [np.array(Image.open(f'{path}/{image}')) for image in image_list]
    )

def get_image_similarity(image_array):
    return np.array([[np.linalg.norm(image_array[i] - image_array[j]) for j in range(len(image_array))] for i in tqdm(range(len(image_array)))])

if __name__ == '__main__':
    path = Path('inputs') / 'valid'
    folders = [folder for folder in os.listdir(path) if os.path.isdir(f'{path}/{folder}')]
    for folder in folders:
        print(f'Processing {folder}...')
        folder_path = path / folder
        image_list = get_image_list(folder_path)
        image_list = image_list[:500]
        image_array = get_image_array(folder_path, image_list)
        image_similarity = get_image_similarity(image_array)
        print(image_similarity)
        print(image_similarity.shape)
        #print max min and mean
        print(f'max: {image_similarity.max()}')
        print(f'min: {image_similarity.min()}')
        print(f'mean: {image_similarity.mean()}')

        plt.figure(figsize=(10, 10))
        plt.imshow(image_similarity)
        plt.show()

        # subplots
        n_rows = 5
        n_cols = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

        #show the most similar images
        i = 0
        for j in range(image_similarity.shape[0]):
            most_similar = np.argsort(image_similarity[j])
            if(image_similarity[j][most_similar[1]] > 25000 or image_similarity[j][most_similar[1]] < 1):
                continue
            # get the most similar images
            axs[i, 0].imshow(image_array[j])
            axs[i, 1].imshow(image_array[most_similar[0]])
            
            # show distance and image name
            axs[i, 0].set_title(f'{image_similarity[j][most_similar[1]]:.2f} {image_list[j]}')
            axs[i, 1].set_title(f'{image_list[most_similar[1]]}')
            
            i += 1
            
            if i >= n_rows:
                break
            
        
        plt.tight_layout()
                
        

        plt.show()

        
