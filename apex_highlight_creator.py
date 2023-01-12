import os

import imageio
import moviepy.editor as mp
import numpy as np
from loguru import logger as log
from PyQt5.QtCore import QObject, pyqtSignal

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

def dump_inputs(rois, name):
    """
    Given the rois, dump them to inputs folder
    """

    for i, subf in rois.items():
        for key, img in subf.items():
            img = img.astype(np.uint8)
            img_path = f"inputs/{key}_{name}_{i}.png"
            if not os.path.exists(img_path):
                imageio.imwrite(img_path, img)

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

class HighlightVideoCreator(QObject):
    progress = pyqtSignal(int)
    
    def __init__(self, folder, filename, rois, roi_size, default_image_size):
        super().__init__()
        
        self.folder = folder
        self.filename = filename
        self.path = os.path.join(folder, filename)
        self.clip = mp.VideoFileClip(self.path)
        self.rois = rois
        self.roi_size = roi_size
        self.default_image_size = default_image_size

    def set_roi_size(self, roi_size):
        self.roi_size = roi_size

    def create_subclip(self, indices):
        """
        Given a clip and the indices of the relavant frames, create a clip only with those frames
        """
        subclips = []

        start = 0
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                # If the current index is not consecutive with the previous index, create a new subclip
                subclips.append(self.clip.subclip(
                    t_start=indices[start] / self.clip.fps, t_end=indices[i-1] / self.clip.fps))
                start = i

        subclips.append(self.clip.subclip(
            t_start=indices[start] / self.clip.fps, t_end=indices[-1] / self.clip.fps))

        subclip = mp.concatenate_videoclips(subclips)

        return subclip

    def create_highlight_video(self, mask, output_path):
        """
        Given a clip and a mask, create a video with only the frames where the mask is 1
        """

        mask_indices = mask.nonzero()[0]

        if len(mask_indices) > 0:
            filtered_clip = self.create_subclip(mask_indices)

            filtered_clip.write_videofile(output_path)

    def get_mask(self, name):
        """
        Given a clip and the subframes, predict if each frame is interesting or not
        returns a mask with 1s where the frame is interesting and 0s where it is not
        """
        self.progress.emit(15)

        self.model, _ = create_model()

        mask = np.zeros(int(self.clip.duration * self.clip.fps))
        for i, subf in self.subframes.items():
            self.progress.emit((i+1.) / len(self.subframes) * 50 + 20)

            pred = []
            for key, subframe in subf.items():
                with self.model.no_mbar():
                    pred_class, pred_idx, outputs = self.model.predict(subframe)
                pred.append(pred_class == 'true')

                if i % 3 == 0:
                    img_path = f"outputs/{pred_class}/{key}_{name[-20:]}_{i/60:.2f}.png"
                    if not os.path.exists(img_path):
                        subframe = subframe.astype(np.uint8)
                        imageio.imwrite(img_path, subframe)
            if i < mask.size:
                mask[i] = any(pred)
        return mask

    
    def get_roi_images(self, rois, roi_size, sampling):
        """
        Given a clip, extract the subframes of the regions of interest
        """
        roi_images = {}
        times = np.arange(0, self.clip.duration, 1.0/self.clip.fps)
        for i, t in enumerate(times):
            self.progress.emit((i+1.) / times.size * 100 * .15)
            if i % sampling != 0:
                continue
            try:
                frame = self.clip.get_frame(t)
                roi_images[i] = {key: get_subframe(
                    frame, position, roi_size) for key, position in rois.items()}
            except Exception:
                continue
        return roi_images

    def create_highlight(self, predict_sampling, keep_before, keep_after):
        """
        Given a clip, create a video with only the interesting frames
        """
        self.progress.emit(0)

        a_rois, a_roi_size = adapt_rois(self.rois, self.roi_size, self.default_image_size, self.clip.size[::-1])

        # Get the subframes of the regions of interest
        self.subframes = self.get_roi_images(a_rois, a_roi_size, predict_sampling)

        log.info(f"{self.filename}: collected {len(self.subframes)} subframes")

        # Predict if each frame is interesting or not
        mask = self.get_mask(self.filename[:-20])

        log.info(f"{self.filename}: selected {mask.sum()} interesting frames")

        # Select the frames around the interesting frames
        mask = expand_ones(mask, keep_before * self.clip.fps, keep_after * self.clip.fps)

        output_path = f"videos/{self.filename[:-4]}_shortened.mp4"
        
        self.progress.emit(80)

        log.info(f"{self.filename}: creating {output_path}")

        # Create the video with only the interesting frames
        self.create_highlight_video(mask, output_path)

        log.info(f"{self.filename}: created {output_path}")
        self.progress.emit(100)

    def generate_input_images(self, input_generation_sampling):
        """
        Given a clip, extract the subframes of the regions of interest and dump them to inputs folder
        """
        
        a_rois, a_roi_size = adapt_rois(self.rois, self.roi_size, self.default_image_size, self.clip.size[::-1])

        subframes = self.get_roi_images(a_rois, a_roi_size, input_generation_sampling)

        dump_inputs(subframes, self.filename[:-20])