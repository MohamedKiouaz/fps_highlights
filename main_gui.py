import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger as log
from PyQt5 import QtWidgets, QtGui

from apex_highlight_creator import (create_folders,
                             HighlightVideoCreator)

class ProgressBarWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1500, 600)
        self.setWindowTitle("Progress Bar")
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        self.layout = QtWidgets.QVBoxLayout(self.widget)
        self.widget.setLayout(self.layout)

    def start_processing(self):
        log.info(f'Working on {folder}.')
        self.executor = ThreadPoolExecutor(max_workers=3)
        for i, filename in enumerate(os.listdir(folder)):
            if not filename.endswith(".mp4"):
                continue
            
            pb = QtWidgets.QProgressBar(self)
            pb.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            
            pb.setFormat(f'{filename} %p%')

            self.layout.addWidget(pb)
            
            log.info(f'Working on {filename}.')
            
            hlvc = HighlightVideoCreator(folder, filename, rois, roi_size, default_image_size)
            hlvc.progress.connect(pb.setValue)

            #hlvc.create_highlight(predict_sampling, keep_before, keep_after)

            if generate_inputs:
                self.executor.submit(hlvc.generate_input_images, input_generation_sampling)
            else:
                self.executor.submit(hlvc.create_highlight, predict_sampling, keep_before, keep_after)

        log.info('Started processes.')
        
        # Show the progress bar window
        self.show()

if __name__ == "__main__":
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
    
    app = QtWidgets.QApplication([])
    window = ProgressBarWindow()
    window.start_processing()
    app.exec_()
