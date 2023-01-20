import os
import shutil

import imageio
import moviepy.editor as mp
import numpy as np
from loguru import logger as log
from tqdm import tqdm
from pathlib import Path

from ml import create_model


def expand_ones(arr, before, after):
    """
    Expand the ones in an array of zeros and ones.

    Given an array `arr` containing zeros and ones, this function expands the ones by replacing
    the elements before and after each one with ones as well. The number of elements to replace
    before and after each one is given by the parameters `N` and `M`, respectively.

    """
    arr = np.array(arr)
    arr = arr == 1

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

def get_subframe(image, position, size):
    """
    Given an input image and a position, extract a subframe of size subframe_size
    """
    rows, cols, _ = image.shape

    frame_rows, frame_cols = size

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
        # If the mask is almost all ones, just copy the original video
        shutil.copyfile(clip.filename, output_path)
        log.info("Video is almost all highlight, copying original video instead of creating a new one")
        return

    mask_indices = mask.nonzero()[0]

    if len(mask_indices) > 0:
        filtered_clip = create_subclip(clip, mask_indices)

        filtered_clip.write_videofile(output_path)

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

class FPSHighlighter:
    """
    A class that highlights the frames of a video where a certain action is performed
    """

    def __init__(self, path):
        self.clip = mp.VideoFileClip(path)
        path = Path(self.clip.filename)
        self.basename = path.stem

    def set_rois(self, rois, roi_size, default_image_size):
        a_rois, a_roi_size = adapt_rois(rois, roi_size, default_image_size, self.clip.size[::-1])
        self.rois = a_rois
        self.roi_size = a_roi_size

    def set_expand_rules(self, keep_before, keep_after):
        self.keep_before = keep_before
        self.keep_after = keep_after

    def set_sampling(self, predict, input_generation=None, output_generation=None):
        self.predict_sampling = predict
        
        if input_generation is None:
            self.input_generation_sampling = 5 * predict
        
        if output_generation is None:
            self.output_generation_sampling = 15 * predict

    def get_roi_images(self):
        """
        Given a clip, extract the subframes of the regions of interest
        """
        roi_images = {}
        times = np.arange(0, self.clip.duration, 1.0/self.clip.fps)
        for i, t in tqdm(enumerate(times), total=times.size):
            if i % self.predict_sampling != 0:
                continue
            try:
                frame = self.clip.get_frame(t)
                roi_images[i] = {key: get_subframe(
                    frame, position, self.roi_size) for key, position in self.rois.items()}
            except Exception:
                continue
        return roi_images

    def get_mask(self):
        """
        Given a clip, extract the rois and predict whether they are highlights or not
        returns a mask of the same size as the clip
        """

        mask, times = self.init_mask()

        for i, t in tqdm(enumerate(times), total=times.size):
            if i % self.predict_sampling != 0:
                continue

            if i > self.keep_before and mask[i-self.keep_before//2:i].max() > 0:
                # No need to predict if this frame is already a neighbor of a highlight
                continue

            if i > 0 and i % (60 * self.predict_sampling) == 0:
                np.save(f"temp/{self.basename}.npy", mask)
            
            try:
                frame = self.clip.get_frame(t)
                
                pred = []
                for key in self.rois:
                    subframe = get_subframe(frame, self.rois[key], self.roi_size)
                    
                    pred_class = self.predict_subframe(subframe)
                    
                    pred.append(pred_class)
                    
                    if i % (15 * self.predict_sampling) == 0:
                        subframe_path = f"outputs/{pred_class}/{key}_{self.basename}_{t:.2f}.png"
                        if not os.path.exists(subframe_path):
                            subframe = subframe.astype(np.uint8)
                            imageio.imwrite(subframe_path, subframe)
                
                if i < mask.size:
                    mask[i] = any(pred)
        
            except Exception:
                continue
        
        np.save(f"temp/{self.basename}.npy", mask)

        return mask

    def init_mask(self):
        """
        Given a clip, initialize the mask and the times of the frames
        """

        mask = None

        # if we have already generated the mask, load it
        if os.path.exists(f"temp/{self.basename}.npy"):
            mask = np.load(f"temp/{self.basename}.npy")

        if mask is None:
            mask = -np.ones(int(self.clip.duration * self.clip.fps))

        # get the times of the frames that need to be predicted
        times = np.arange(0, self.clip.duration, 1.0/self.clip.fps)
        
        # get the last index that was already predicted
        where = np.where(mask != -1)
        if len(where[0]) > 0:
            last_index = np.where(mask != -1)[0][-1]
            log.info(f"Recovered from previous run: {last_index/mask.size*100:.2f}%.")
            duration = expand_ones(mask, self.keep_before, self.keep_after).sum()
            log.info(f"Recovered highlight duration: {duration/self.clip.fps:.2f}s.")
            times = times[last_index:]

        return mask, times

    def predict_subframe(self, subframe):
        """
        Given a subframe, predict whether it is a highlight or not
        """
        with self.model.no_mbar():
            pred_class, pred_idx, outputs = self.model.predict(subframe)

        return pred_class

    def generate_inputs(self):
        """
        Given a clip, extract the subframes of the regions of interest and dump them to inputs folder
        """
        
        subframes = self.get_roi_images()

        dump_inputs(subframes, self.basename)

    def create_highlight(self):
        """
        Given a clip, create a video with only the interesting frames
        """
        # We need to load the model to make predictions
        self.model, _ = create_model()
        
        # Get the subframes of the regions of interest
        mask = self.get_mask()

        # Save the mask to temp folder in case we need to resume
        np.save(f"temp/{self.basename}.npy", mask)

        # Select the frames around the interesting frames
        mask = expand_ones(mask, self.keep_before * self.clip.fps, self.keep_after * self.clip.fps)

        # Create the video with only the interesting frames
        output_path = f"videos/{self.basename}_shortened.mp4"
        
        create_highlight_video(self.clip, mask, output_path)