import os

import moviepy.editor as mp
from loguru import logger as log

from apex_highlights import (create_highlight,
                                    generate_inputs_from_image,
                                    create_folders,
                                    adapt_rois)

if __name__ == '__main__':
    # if True, generate inputs from the videos
    # if False, use the inputs to predict
    # you need to generate the inputs only once
    # then sort them in the inputs folder
    # then you run ml.py
    # then you can use the model to predict
    generate_inputs = False

    # folder where the videos are
    folder = r"D:\Videos\Radeon ReLive\Apex Legends\a"

    # if in input generation, generate one input every
    # this number needs to be high because we have to see a variaty of situations
    # many guns, many champions, many maps, many gamemodes...
    input_generation_sampling = 50

    # if in prediction mode, use one frame every
    # this number needs to be low because we want to be able to detect the interesting frames
    # if too high, we will miss some interesting frames
    predict_sampling = 5

    # keep this number of seconds before and after an interesting frame
    keep_after = 5  # sec
    keep_before = 4  # sec

    # size of the regions of interest
    roi_size = (100, 100)

    # name and positions of the regions of interest
    rois = {'center': (719, 1289), 'teamsleft': (88, 2049)}

    # default image size
    default_image_size = (1440, 2560)

    create_folders()

    log.info(f'Working on {folder}.')
    for filename in os.listdir(folder):
        if not filename.endswith(".mp4"):
            continue
        
        log.info(f'Working on {filename}.')

        path = os.path.join(folder, filename)

        with mp.VideoFileClip(path) as clip:
            a_rois, a_roi_size = adapt_rois(rois, roi_size, default_image_size, clip.size[::-1])

            if generate_inputs:
                generate_inputs_from_image(clip, filename, a_rois, a_roi_size, input_generation_sampling)
            else:
                create_highlight(clip, filename, predict_sampling, keep_before, keep_after, a_rois, a_roi_size)
