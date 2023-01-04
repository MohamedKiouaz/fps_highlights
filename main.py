import moviepy.editor as mp
import os
import numpy as np
from tqdm import tqdm
import imageio
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner

def expand_ones(arr, N, M):
    """
    Expand the ones in an array of zeros and ones.
    
    Given an array `arr` containing zeros and ones, this function expands the ones by replacing
    the elements before and after each one with ones as well. The number of elements to replace
    before and after each one is given by the parameters `N` and `M`, respectively.
    
    """
    arr = np.array(arr)

    # Find the indices of the zeros in the array
    N = int(N)
    M = int(M)
    print(arr.sum() / arr.size)
    zero_indices = np.where(arr == 1)[0]

    # Replace the elements before and after each zero with zeros
    for i in zero_indices:
        start = max(0, i - N)
        end = min(len(arr), i + M + 1)
        arr[start:end] = 1
    print(arr.sum() / arr.size)
    return arr

def get_subframe(image, position, subframe_size):
    """
    Given an input image and a position, extract a subframe of size subframe_size
    """
    rows, cols, _ = image.shape

    frame_rows, frame_cols = subframe_size

    row_min = position[0] - frame_rows // 2
    col_min = position[1] - frame_cols // 2

    row_min = max(row_min, 0)
    col_min = max(col_min, 0)
    if row_min + frame_rows > rows:
        row_min = rows - frame_rows
    if col_min + frame_cols > cols:
        col_min = cols - frame_cols

    return image[row_min:row_min+frame_rows, col_min:col_min+frame_cols]

def create_subclip(clip, indices):
    """
    Given a clip and the indices of the relavant frames, create a clip only with those frames
    """
    subclips = []

    start = 0
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            # If the current index is not consecutive with the previous index, create a new subclip
            subclips.append(clip.subclip(t_start=indices[start] / clip.fps, t_end=indices[i-1] / clip.fps))
            start = i

    subclips.append(clip.subclip(t_start=indices[start] / clip.fps, t_end=indices[-1] / clip.fps))
    
    subclip = mp.concatenate_videoclips(subclips)

    return subclip

if __name__ == '__main__':
    generate_inputs = False

    # folder where the videos are
    folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"

    # if in input generation, generate one input every 
    # this number needs to be high because we have to see a variaty of situations
    # many guns, many champions, many maps...
    input_generation_sampling = 50

    # if in prediction mode, use one frame every
    # this number needs to be low because we want to be able to detect the interesting frames
    # if too high, we will miss some interesting frames
    predict_sampling = 5

    # keep this number of seconds before and after an interesting frame
    keep_after = 5 # sec
    keep_before = 4 # sec
    
    if not generate_inputs:
            # We need to load the model to make prediction if a frame is interesting or not
            path = Path('inputs')
            data = ImageDataLoaders.from_folder(path, train='train', valid='valid')
            learn = vision_learner(data, models.resnet18, bn_final=True, model_dir="models")
            learn = learn.load('resnet18')

    for filename in os.listdir(folder):
        if not filename.endswith(".mp4"):
            continue

        path = os.path.join(folder, filename)
        print(f'Working on {path}.')
        clip = mp.VideoFileClip(path)

        times = np.arange(0, clip.duration, 1.0/clip.fps)

        subframes = {}
        for i, t in tqdm(enumerate(times), total = times.size):
            if i % predict_sampling != 0:
                continue
            try:
                frame = clip.get_frame(t)
                subframes[i] = {'center': get_subframe(frame, (719, 1289), (100, 100)),
                        'teamsleft': get_subframe(frame, (88, 2049), (100, 100))}
            except Exception:
                continue

        mask = np.zeros(int(clip.duration * clip.fps))
        for i, subf in tqdm(subframes.items()):
            pred = []
            for key, subframe in subf.items():
                subframe = subframe.astype(np.uint8)
                if not generate_inputs:
                    with learn.no_bar():
                        pred_class, pred_idx, outputs = learn.predict(subframe)
                    pred.append(pred_class == 'true')
                    if i%5 == 0:
                        img_path = f"outputs/{pred_class}/{key}_{filename[-20:]}_{i/60:.2f}.png"
                        if not os.path.exists(img_path):
                            imageio.imwrite(img_path, subframe)
                else:
                    img_path = f"inputs/{key}_{filename[-20:]}_{i/60:.2f}.png"
                    if not os.path.exists(img_path):
                        imageio.imwrite(img_path, subframe)

            mask[i] = any(pred)

        if not generate_inputs:
            mask = expand_ones(mask, keep_before * clip.fps, keep_after * clip.fps)

            mask_indices = mask.nonzero()[0]

            if len(mask_indices) > 0:
                filtered_clip = create_subclip(clip, mask_indices)

                filtered_clip.write_videofile(f"videos/{filename[:-4]}_shortened.mp4")