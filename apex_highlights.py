import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import imageio
import moviepy.editor as mp
import numpy as np
from tqdm import tqdm
from loguru import logger as log

from ml import create_model


def expand_ones(arr, before, after):
    """
    Expand the ones in an array of zeros and ones.

    Given an array `arr` containing zeros and ones, this function expands the ones by replacing
    the elements before and after each one with ones as well. The number of elements to replace
    before and after each one is given by the parameters `N` and `M`, respectively.

    """
    arr = np.array(arr)

    # Find the indices of the zeros in the array
    before = int(before)
    after = int(after)

    orig_percent = arr.sum() / arr.size

    zero_indices = np.where(arr == 1)[0]

    # Replace the elements before and after each zero with zeros
    for i in zero_indices:
        start = max(0, i - before)
        end = min(len(arr), i + after + 1)
        arr[start:end] = 1

    log.info(f"{orig_percent*100:.4f}% -> {arr.sum() / arr.size * 100:.4f}%")
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
            subclips.append(clip.subclip(
                t_start=indices[start] / clip.fps, t_end=indices[i-1] / clip.fps))
            start = i

    subclips.append(clip.subclip(
        t_start=indices[start] / clip.fps, t_end=indices[-1] / clip.fps))

    subclip = mp.concatenate_videoclips(subclips)

    return subclip


def create_highlight_video(clip, mask, output_path):
    """
    Given a clip and a mask, create a video with only the frames where the mask is 1
    """

    if mask.sum()/mask.size >= .99:
        shutil.copyfile(clip.filename, output_path)
        return

    mask_indices = mask.nonzero()[0]

    if len(mask_indices) > 0:
        filtered_clip = create_subclip(clip, mask_indices)

        filtered_clip.write_videofile(output_path)


def get_roi_images(clip, rois, roi_size, sampling):
    """
    Given a clip, extract the subframes of the regions of interest
    """
    roi_images = {}
    times = np.arange(0, clip.duration, 1.0/clip.fps)
    for i, t in tqdm(enumerate(times), total=times.size):
        if i % sampling != 0:
            continue
        try:
            frame = clip.get_frame(t)
            roi_images[i] = {key: get_subframe(
                frame, position, roi_size) for key, position in rois.items()}
        except Exception:
            continue
    return roi_images


def get_mask(clip, subframes, model, name):
    """
    Given a clip and the subframes, predict if each frame is interesting or not
    returns a mask with 1s where the frame is interesting and 0s where it is not
    """
    mask = np.zeros(int(clip.duration * clip.fps))
    for i, subf in tqdm(subframes.items()):
        pred = []
        for key, subframe in subf.items():
            with model.no_mbar():
                pred_class, pred_idx, outputs = model.predict(subframe)
            pred.append(pred_class == 'true')

            if i % 3 == 0:
                img_path = f"outputs/{pred_class}/{key}_{name}_{i/60:.2f}.png"
                if not os.path.exists(img_path):
                    subframe = subframe.astype(np.uint8)
                    imageio.imwrite(img_path, subframe)
        if i < mask.size:
            mask[i] = any(pred)
    return mask


def dump_inputs(rois, name):
    """
    Given the rois, dump them to inputs folder
    """

    for i, subf in tqdm(rois.items()):
        for key, img in subf.items():
            img = img.astype(np.uint8)
            img_path = f"inputs/{key}_{name}_{i}.png"
            if not os.path.exists(img_path):
                imageio.imwrite(img_path, img)


def generate_inputs_from_image(clip, filename, rois, roi_size, input_generation_sampling):
    """
    Given a clip, extract the subframes of the regions of interest and dump them to inputs folder
    """
    
    subframes = get_roi_images(clip, rois, roi_size, input_generation_sampling)

    dump_inputs(subframes, filename[:-20])


def create_highlight(clip, filename, predict_sampling, keep_before, keep_after, rois, roi_size):
    """
    Given a clip, create a video with only the interesting frames
    """

    # Get the subframes of the regions of interest
    subframes = get_roi_images(clip, rois, roi_size, predict_sampling)

    # We need to load the model to make predictions
    model, _ = create_model()

    # Predict if each frame is interesting or not
    mask = get_mask(clip, subframes, model, filename[:-20])

    # Select the frames around the interesting frames
    mask = expand_ones(mask, keep_before * clip.fps, keep_after * clip.fps)

    output_path = f"videos/{filename[:-4]}_shortened.mp4"

    
    # Create the video with only the interesting frames
    create_highlight_video(clip, mask, output_path)


def create_folders():
    """
    Create the folders where we will store the inputs and outputs
    """
    for folder in ['inputs/train/true', 'inputs/train/false', 'inputs/valid/true', 'inputs/valid/false', 'outputs/true', 'outputs/false', 'videos']:
        os.makedirs(folder, exist_ok=True)

def adapt_rois(rois, roi_size, base_image_size, final_image_size):
    """
    Adapt the rois to the final image size
    """
    adapted_rois = {
        key: (
            int(roi[0] * final_image_size[0] / base_image_size[0]),
            int(roi[1] * final_image_size[1] / base_image_size[1]),
        )
        for key, roi in rois.items()
    }

    adapted_roi_size = (int(roi_size[0] * final_image_size[0] / base_image_size[0]), 
    int(roi_size[1] * final_image_size[1] / base_image_size[1]))
    
    return adapted_rois, adapted_roi_size