# Apex Legends Filtering Script

This script is designed to filter out interesting frames in an Apex Legends video record and save them to a new mp4 file. Interesting frames are defined as those where the player is dealing damage or near the champion. The interesting frames and the ones immediately before or after them are written to the filtered clip.

## Fast Start
- Install Python
- Install the dependencies
- In `main.py`, set `generate_inputs=True`. This will generate the inputs for the machine learning model.
- Sort the inputs in the `inputs` folder into 2 folders: `true` and `false`. The `true` folder contains the interesting frames, and the `false` folder contains the not interesting frames. You need to get at least 2k images in total.
- Train the machine learning model using `ml.py`. This should not take too long.
- In `main.py`, set `generate_inputs=False`. This will use the machine learning model to filter the video.
- In `main.py`, set the `folder` variable to the path to the directory containing the video files to be filtered.

## Dependencies

Use the following command to install the dependencies:

`pip install -r requirements.txt`

## Usage

To use the script, modify the following variables at the bottom of the script:

- `folder`: the path to the directory containing the video files to be filtered
- `generate_inputs`: set to True to generate inputs for machine learning, False to filter the video using the machine learning model
- `input_generation_sampling`: if in input generation mode, this variable determines the frequency at which inputs are generated (one input every input_generation_sampling frames)
- `predict_sampling`: if in prediction mode, this variable determines the frequency at which frames are predicted (one prediction every predict_sampling frames). It seems like hit markers are visible for 15 or so frames. You will probably want to set this variable to less.
- `keep_after`: the number of seconds to keep after an interesting frame
- `keep_before`: the number of seconds to keep before an interesting frame

Once the variables are set, run the script using the following command:

`python main.py`

The script will process each mp4 file in the specified directory and save the filtered clip as a new mp4 file with the same name as the original file.

## Machine Learning

The machine learning aspect of the script is handled in the `ml.py` file. This file contains the code for training and evaluating the machine learning model used to classify the frames as interesting or not. The specific implementation details of the machine learning model are not provided in this script.

## Dataset

You have to sort the pictures into 2 datasets. The proposed classes are :
- `true` : interesting frames
  - hit markers
  - 3 teams left
  - victory
- `false` : not interesting frames
  - nothing happens in the game
  - every screenshot that isn't Apex Legends

You need a minimum of 2k images in total.
Make sure to have diverse images in your dataset:
- different backgrounds: sky, buildings, explosions, etc.
- different weapons: snipers, shotguns, pistols, etc.
- different champions and their abilities
- if you choose to include ramaining teams counter, you need to have all of them (the 19).

### Examples of interesting images

![interesting image 1](images/interesting1.png)
![interesting image 2](images/interesting2.png)
![interesting image 3](images/interesting3.png)
![interesting image 4](images/interesting4.png)
![interesting image 5](images/interesting5.png)
![interesting image 6](images/interesting6.png)

### Examples of not interesting images

![Not interesting image 1](images/not_interesting1.png)
![Not interesting image 2](images/not_interesting2.png)
![Not interesting image 3](images/not_interesting3.png)
![Not interesting image 4](images/not_interesting4.png)
![Not interesting image 5](images/not_interesting5.png)
![Not interesting image 6](images/not_interesting6.png)