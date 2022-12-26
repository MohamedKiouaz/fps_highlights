import moviepy.editor as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import fastai
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner


folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"


def expand_zeros(arr, N, M):
    # Find the indices of the zeros in the array
    N = int(N)
    M = int(M)
    print(arr.sum() / 44100)
    zero_indices = np.where(arr == 1)[0]

    # Replace the elements before and after each zero with zeros
    for i in zero_indices:
        start = max(0, i - N)
        end = min(len(arr), i + M + 1)
        arr[start:end] = 1
    print(arr.sum() / 44100)
    return arr

def get_frame(image, position, frame_size):
    rows, cols, _ = image.shape

    frame_rows, frame_cols = frame_size

    # Calculate top-left corner of frame
    row_min = position[0] - frame_rows // 2
    col_min = position[1] - frame_cols // 2

    if row_min < 0:
        row_min = 0
    if col_min < 0:
        col_min = 0
    if row_min + frame_rows > rows:
        row_min = rows - frame_rows
    if col_min + frame_cols > cols:
        col_min = cols - frame_cols

    frame = image[row_min:row_min+frame_rows, col_min:col_min+frame_cols]

    return frame

generate_inputs = False

predict_sampling = 25

for filename in os.listdir(folder):
    if filename.endswith(".mp4"):
        path = os.path.join(folder, filename)
        print(path)
        clip = mp.VideoFileClip(path)
        audio = mp.AudioFileClip(path)

        if not generate_inputs:
            path = Path('inputs')
            data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2)
            input_shape = data.one_batch()[0].shape[1:]
            learn = vision_learner(data, models.resnet50, bn_final=True, model_dir="models")
            learn = learn.load('model')
        
        mask = []
        i = 0
        for frame in tqdm(clip.iter_frames(), total=int(clip.duration * clip.fps)):
            i += 1
            if generate_inputs and i % 50 != 0:
                continue
            
            if not generate_inputs and i % predict_sampling != 0:
                mask.append(False)
                continue

            subf = {'center': get_frame(frame, (719, 1289), (100, 100)),
                    'teamsleft': get_frame(frame, (88, 2049), (100, 100))}
            pred = []
            for key, subframe in subf.items():
                subframe = subframe.astype(np.uint8)
                if not generate_inputs:
                    pred_class, pred_idx, outputs = learn.predict(subframe)
                    pred.append(pred_class == 'true')
                    imageio.imwrite(f"outputs/{pred_class}/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)
                else:
                    imageio.imwrite(f"inputs/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)

            mask.append(any(pred))

        if not generate_inputs:
            mask = np.array(mask)

            mask = expand_zeros(mask, 4 * clip.fps, 1 * clip.fps)

            mask_indices = mask.nonzero()[0]

            subclips = []

            start = 0
            for i in range(1, len(mask_indices)):
                if mask_indices[i] != mask_indices[i-1] + 1:
                    # If the current index is not consecutive with the previous index, create a new subclip
                    subclips.append(clip.subclip(t_start=mask_indices[start] / clip.fps, t_end=mask_indices[i-1] / clip.fps))
                    start = i

            subclips.append(clip.subclip(t_start=mask_indices[start] / clip.fps, t_end=mask_indices[-1] / clip.fps))
            
            filtered_clip = mp.concatenate_videoclips(subclips)

            filtered_clip.write_videofile(f"videos/{filename[:-4]}_shortened.mp4")
