import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio
import moviepy.editor as mp
import numpy as np
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from tqdm import tqdm
from loguru import logger as log


def crop_video_promise(filepath_original, window, filepath_output):
    command = f"C:/Users/moham/Downloads/HandBrakeCLI-1.6.0-win-x86_64/HandBrakeCLI.exe -i \"{filepath_original}\" -o \"{filepath_output}\" --crop {':'.join([str(x) for x in window])}"
    command += " --encoder-preset veryfast"
    command += " --audio none"
    command += " --rate 1.5"
    command += " --stop-at duration:600"
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
    log.info(arr.sum() / arr.size)
    zero_indices = np.where(arr == 1)[0]

    # Replace the elements before and after each zero with zeros
    for i in zero_indices:
        start = max(0, i - N)
        end = min(len(arr), i + M + 1)
        arr[start:end] = 1
    log.info(arr.sum() / arr.size)
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

class VideoHighlightProcessor:
    def __init__(self, folder, generate_inputs=False, input_generation_sampling=50, predict_sampling=20, keep_after=5, keep_before=4):
        self.folder = folder
        self.generate_inputs = generate_inputs
        self.input_generation_sampling = input_generation_sampling
        self.predict_sampling = predict_sampling
        self.keep_after = keep_after
        self.keep_before = keep_before
        self.multi_thread = False
        self.window_size = (100, 100)
        self.positions = {'center': (719, 1289), 'teams_left': (88, 2049)}
        
        self.temp_folder = 'temp'
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        for filename in os.listdir(self.temp_folder):
            os.remove(os.path.join(self.temp_folder, filename))

        if self.multi_thread:
            self.executor = ThreadPoolExecutor()

        if not self.generate_inputs:
            # We need to load the model to make prediction if a frame is intresting or not
            log.info("Loading model...")
            path = Path('inputs')
            data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2)
            self.learn = vision_learner(data, models.resnet18, bn_final=True, model_dir="models")
            self.learn = self.learn.load('resnet18')
            log.info("Model loaded.")

    def enable_multi_thread(self):
        self.multi_thread = True
        self.executor = ThreadPoolExecutor()

    def routine(self):
        for filename in os.listdir(folder):
            if not filename.endswith(".mp4"):
                continue

            path = os.path.join(folder, filename)
            log.info(f'Working on {path}.')
            vfc = mp.VideoFileClip(path)
            vfc_size = vfc.size
            vfc_duration = vfc.duration
            vfc_fps = vfc.fps
            vfc.close()
            cropped_videos = self.crop_video(filename, clip_size=vfc_size)

            mask = np.zeros((int(vfc.duration * vfc.fps), len(self.positions)), dtype=np.bool)
            for cropped_video in cropped_videos:
                preds = self.process(mp.VideoFileClip(cropped_video['filepath_output']), cropped_video['filepath_output'], cropped_video['key'])
                for timestamp, pred in preds.items():
                    mask[int(timestamp*vfc_fps)] = pred

            if not self.generate_inputs:
                vfc = mp.VideoFileClip(path)
                self.generate_video(vfc, mask, filename)

    def crop_video(self, filename, clip_size):
        log.info(f"Cropping {filename}")

        filepath_original = os.path.join(self.folder, filename)
        promises = []
        for key, position in self.positions.items():
            filepath_output = os.path.join(self.temp_folder, f"cropped_{key}_{filename}")
            window = [clip_size[1] - (position[0] + self.window_size[0] // 2), position[0] - self.window_size[0] // 2,
                    position[1] - self.window_size[1] // 2, clip_size[0] - (position[1] + self.window_size[1] // 2)]
            d = {
                "key":
                key,
                "position":
                position,
                "filepath_output":
                filepath_output,
                "promise":
                crop_video_promise(filepath_original, window,
                                    filepath_output),
            }
            promises.append(d)

        for d in promises:
            output, error = d['promise'].communicate()
            log.info(f"Finished cropping {filename} around {d['key']}")

        return promises
    
    def process(self, clip, filename, key):
        log.info(f"Processing {filename} around {key}.")
        preds = {}
        self.tqdm = tqdm(total = int(clip.duration * clip.fps))
        for i, subframe in enumerate(clip.iter_frames()):
            self.tqdm.update(1)
            
            if i % self.predict_sampling != 0:
                continue
            
            pred = []
            subframe = subframe.astype(np.uint8)
            if not self.generate_inputs:
                with self.learn.no_bar():
                    pred_class, pred_idx, outputs = self.learn.predict(subframe)
                pred.append(pred_class == 'true')
                if i % 17 == 0:
                    imageio.imwrite(f"outputs/{pred_class}/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)
            else:
                imageio.imwrite(f"inputs/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)
            
            preds[float(i) / clip.fps] = any(pred)
        self.tqdm.close()
        return preds

    def generate_video(self, clip, mask, filename):        
        mask = expand_ones(mask, self.keep_before * clip.fps, self.keep_after * clip.fps)

        mask_indices = mask.nonzero()[0]
        
        if len(mask_indices) < 0:
            log.warning(f"Could not find any interesting frames in {filename}.")

            return
        
        filtered_clip = create_subclip(clip, mask_indices)

        filtered_clip.write_videofile(f"videos/{filename[:-4]}_shortened.mp4")

if __name__ == '__main__':
    folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"
    generate_inputs = False
    input_generation_sampling = 50 # if in input generation, generate one input every 
    predict_sampling = 20 # if in prediction mode, use one frame every
    keep_after = 5 # sec
    keep_before = 4 # sec

    vp = VideoHighlightProcessor(folder, generate_inputs, input_generation_sampling, predict_sampling, keep_after, keep_before)
    vp.routine()