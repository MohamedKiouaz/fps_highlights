import os


def create_folders():
    """
    Create the folders where we will store the inputs and outputs
    """
    for folder in ['inputs/models', 'inputs/train/true', 'inputs/train/false', 'inputs/valid/true', 'inputs/valid/false', 'outputs/true', 'outputs/false', 'videos', 'temp', 'low_confidence/true', 'low_confidence/false']:
        os.makedirs(folder, exist_ok=True)
