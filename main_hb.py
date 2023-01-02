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


def crop_video_promise(filepath_original, window, filepath_output):
  command = f"HandBrakeCLI -i {filepath_original} -o {filepath_output} --crop {window[0]}:{window[1]}:{window[2]}:{window[3]}"
  return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)


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

class VideoProcessor:
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

        if self.multi_thread:
            self.executor = ThreadPoolExecutor()

        if not self.generate_inputs:
            # We need to load the model to make prediction if a frame is intresting or not
            print("Loading model...")
            path = Path('inputs')
            data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2)
            self.learn = vision_learner(data, models.resnet18, bn_final=True, model_dir="models")
            self.learn = self.learn.load('resnet18')
        

    def routine(self):
        for filename in os.listdir(folder):
            if not filename.endswith(".mp4"):
                continue

            path = os.path.join(folder, filename)
            print(f'Working on {path}.')
            vfc = mp.VideoFileClip(path)
            self.tqdm = tqdm()
            self.tqdm.total = int(vfc.duration * vfc.fps)
            
            cropped_videos = self.crop_video(filename)

            mask = np.zeros((int(vfc.duration * vfc.fps), len(self.positions)), dtype=np.bool)
            for cropped_video in cropped_videos:
                mask |= self.shorten(mp.VideoFileClip(cropped_video['filepath_output']), cropped_video['filepath_output'], cropped_video['key'])

            if not self.generate_inputs:
                self.generate_video(vfc, mask, filename)

    def crop_video(self, filename):
        print(f"Cropping {filename}")

        filepath_original = os.path.join(self.folder, filename)
        promises = []
        for key, position in enumerate(self.positions):
            filepath_output = os.path.join(self.folder, f"cropped_{i}_{filename}")
            window = [position[0] - self.window_size[0] // 2, position[0] + self.window_size[0] // 2,
                    position[1] - self.window_size[1] // 2, position[1] + self.window_size[1] // 2]
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
            d["promise"].wait()
            print(f"Finished cropping {d['filename']} around {d['key']}")
        
        return promises
    
    def shorten(self, clip, filename, key):
        #use divide and conquer to shorten the video
        if clip.duration <= self.predict_sampling / clip.fps:
            self.tqdm.update(int(clip.duration * clip.fps))
            return self.process(clip, filename, key)
        
        mid = clip.duration / 2

        m1 = self.shorten(clip.subclip(t_start=0, t_end=mid), filename, key)
        m2 = self.shorten(clip.subclip(t_start=mid, t_end=clip.duration), filename, key)

        return np.concatenate((m1, m2))

    def process(self, clip, filename, key):
        mask = np.zeros(int(clip.duration * clip.fps))
        for i, subframe in enumerate(clip.iter_frames()):
            pred = []
            subframe = subframe.astype(np.uint8)
            if not self.generate_inputs:
                with self.learn.no_bar():
                    pred_class, pred_idx, outputs = self.learn.predict(subframe)
                pred.append(pred_class == 'true')
                if i%17 == 0:
                    imageio.imwrite(f"outputs/{pred_class}/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)
            else:
                imageio.imwrite(f"inputs/{key}_{filename[-20:]}_{i/60:.2f}.png", subframe)

            mask[i] = any(pred)
        
        return mask

    def generate_video(self, clip, mask, filename):        
        mask = expand_ones(mask, self.keep_before * clip.fps, self.keep_after * clip.fps)

        mask_indices = mask.nonzero()[0]
        
        if len(mask_indices) > 0:
            filtered_clip = create_subclip(clip, mask_indices)

            filtered_clip.write_videofile(f"videos/{filename[:-4]}_shortened.mp4")

if __name__ == '__main__':
    folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"
    generate_inputs = False
    input_generation_sampling = 50 # if in input generation, generate one input every 
    predict_sampling = 20 # if in prediction mode, use one frame every
    keep_after = 5 # sec
    keep_before = 4 # sec

    vp = VideoProcessor(folder, generate_inputs, input_generation_sampling, predict_sampling, keep_after, keep_before)
    vp.routine()