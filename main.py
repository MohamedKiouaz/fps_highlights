import os
import glob

from loguru import logger as log

from fps_functions import create_folders

from FPSHighlighter import FPSHighlighter

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
    predict_sampling = 60

    # keep this number of seconds before and after an interesting frame
    keep_after = 3  # sec
    keep_before = 3  # sec

    # default image size
    default_image_size = (1440, 2560)

    # size of the regions of interest
    roi_size = (100, 100)

    # name and positions of the regions of interest
    # must be in the default image size
    rois = {'teamsleft': (88, 2049), 'dammage': (632, 1322), 'hitmarker': (720, 1280), 'shield': (975, 2145)}

    create_folders()

    log.info(f'Working on {folder}.')
    entities = glob.glob(f'{folder}/*.mp4')
    
    for file_number, path in enumerate(entities):
        log.info(f'Working on ({file_number + 1}/{len(entities)}) {path}.')

        highlighter = FPSHighlighter(path)
        highlighter.set_expand_rules(keep_before, keep_after)
        highlighter.set_sampling(predict_sampling, input_generation_sampling)
        highlighter.set_rois(rois, roi_size, default_image_size)

        if generate_inputs:
            highlighter.generate_inputs()
        else:
            highlighter.create_highlight()
