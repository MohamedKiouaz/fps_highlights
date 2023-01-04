import os
import subprocess

import imageio
import moviepy.editor as mp
import numpy as np
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from tqdm import tqdm
from loguru import logger as log


def crop_video_promise(filepath_original, window, filepath_output):
    command = "C:/Users/moham/Downloads/HandBrakeCLI-1.6.0-win-x86_64/HandBrakeCLI.exe"
    command += f" -i \"{filepath_original}\" -o \"{filepath_output}\" --crop {':'.join([str(x) for x in window])}"
    command += " --encoder-preset veryfast"
    command += " --audio none"
    command += " --rate 2.5"
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

    percent = arr.sum() / arr.size
    true_indices = np.where(arr)[0]

    # Replace the elements before and after each zero with zeros
    for i in true_indices:
        start = max(0, i - N)
        end = min(len(arr), i + M + 1)
        arr[start:end] = 1
    
    log.info(f"{percent*100:.4f}% -> {arr.sum() / arr.size * 100:.4f}%")

    return arr

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
    def __init__(self, folder, input_generation_sampling=50, predict_sampling=20, keep_after=5, keep_before=4):
        self.folder = folder
        self.input_generation_sampling = input_generation_sampling
        self.predict_sampling = predict_sampling
        self.keep_after = keep_after
        self.keep_before = keep_before
        self.window_size = (100, 100)
        self.positions = {'center': (719, 1289), 'teams_left': (88, 2049)}
        
        self.temp_folder = 'temp'
        
        self._init_folders()

        for filename in os.listdir(self.temp_folder):
            os.remove(os.path.join(self.temp_folder, filename))

    def _init_folders(self):
        for folder in [self.folder, self.temp_folder, 'inputs/train/true', 'inputs/train/false', 'inputs/valid/true', 'inputs/valid/false', 'outputs/true', 'outputs/false', 'videos']:
            os.makedirs(folder, exist_ok=True)

    def load_model(self):
        # We need to load the model to make prediction if a frame is intresting or not
        log.info("Loading model...")
        path = Path('inputs')
        data = ImageDataLoaders.from_folder(path, train='train', valid='valid')
        self.learn = vision_learner(data, models.resnet18, bn_final=True, model_dir="models")
        self.learn = self.learn.load('resnet18')
        log.info("Model loaded.")

    def highlight(self):
        self.load_model()

        for filename in os.listdir(self.folder):
            if not filename.endswith(".mp4"):
                continue

            path = os.path.join(self.folder, filename)
            log.info(f'Working on {path}.')
            vfc = mp.VideoFileClip(path)
            vfc_size = vfc.size
            vfc_fps = vfc.fps
            vfc.close()
            
            cropped_videos = self.crop_video(filename, clip_size=vfc_size)

            mask = np.zeros(int(vfc.duration * vfc.fps), dtype=np.bool)
            for cropped_video in cropped_videos:
                preds = self.process(mp.VideoFileClip(cropped_video['filepath_output']), cropped_video['filepath_output'], cropped_video['key'])
                for timestamp, pred in preds.items():
                    ind = min(int(timestamp*vfc_fps), mask.size-1)
                    mask[ind] = mask[ind] or pred

            log.info(f"Found {mask.sum()} interesting frames.")

            vfc = mp.VideoFileClip(path)
            self.generate_video(vfc, mask, filename)

    def generate_inputs(self):
        for filename in os.listdir(self.folder):
            if not filename.endswith(".mp4"):
                continue

            path = os.path.join(self.folder, filename)
            log.info(f'Working on {path}.')
            vfc = mp.VideoFileClip(path)
            vfc_size = vfc.size
            vfc_fps = vfc.fps
            vfc.close()
            
            cropped_videos = self.crop_video(filename, clip_size=vfc_size)
        
            for cropped_video in cropped_videos:
                self.dump_inputs(mp.VideoFileClip(cropped_video['filepath_output']), cropped_video['filepath_output'], cropped_video['key'])

    def crop_video(self, filename, clip_size):
        log.info(f"Cropping {filename}")

        filepath_original = os.path.join(self.folder, filename)
        promises = []
        for key, position in self.positions.items():
            filepath_output = os.path.join(self.temp_folder, f"cropped_{key}_{filename}")
            window = [position[0] - self.window_size[0] // 2, clip_size[1] - (position[0] + self.window_size[0] // 2),
                    position[1] - self.window_size[1] // 2, clip_size[0] - (position[1] + self.window_size[1] // 2)]
            log.info(f"Cropping {filename} around {key} with window {window}")
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
        for i, subframe in tqdm(enumerate(clip.iter_frames()), total = int(clip.duration * clip.fps)):
            subframe = subframe.astype(np.uint8)
            with self.learn.no_bar():
                pred_class, pred_idx, outputs = self.learn.predict(subframe)
            if i % 4 == 0:
                imageio.imwrite(f"outputs/{pred_class}/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)
            preds[float(i) / clip.fps] = pred_class == 'true'
        
        return preds

    def dump_inputs(self, clip, filename, key):
        log.info(f"Writing input images of {filename} around {key}.")
        for i, subframe in tqdm(enumerate(clip.iter_frames()), total = int(clip.duration * clip.fps)):
            subframe = subframe.astype(np.uint8)
            imageio.imwrite(f"inputs/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe.astype(np.uint8))

    def generate_video(self, clip, mask, filename):        
        mask = expand_ones(mask, self.keep_before * clip.fps, self.keep_after * clip.fps)

        mask_indices = mask.nonzero()[0]
        
        if len(mask_indices) == 0:
            log.warning(f"Could not find any interesting frames in {filename}.")

            return
        
        filtered_clip = create_subclip(clip, mask_indices)

        filtered_clip.write_videofile(f"videos/{filename[:-4]}_shortened.mp4")

if __name__ == '__main__':
    # folder where the videos are
    folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"

    # if in input generation, generate one input every 
    # this number needs to be high because we have to see a variaty of situations
    # many guns, many champions, many maps...
    input_generation_sampling = 50
    
    # if in prediction mode, use one frame every
    # this number needs to be low because we want to be able to detect the interesting frames
    # if too high, we will miss some interesting frames
    predict_sampling = 20

    # keep this number of seconds before and after an interesting frame
    keep_after = 5 # sec
    keep_before = 4 # sec

    vp = VideoHighlightProcessor(folder, input_generation_sampling, predict_sampling, keep_after, keep_before)
    
    generate_inputs = False

    if generate_inputs:
        # generate inputs for the model
        # this needs to be done at least once
        # you need to sort the inputs into folders
        vp.generate_inputs()
    else:
        # generate the highlights
        vp.highlight()