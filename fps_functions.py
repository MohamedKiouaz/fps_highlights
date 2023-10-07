import os


def _class_folders(folder):
    """
    Returns the folders where we will store the images
    """
    return [os.path.join(folder, 'true'), os.path.join(folder, 'false'), os.path.join(folder, 'rank'), os.path.join(folder, 'shield')]

def create_folders():
    """
    Create the folders where we will store the inputs and outputs
    """
    folders = ['inputs/train', 'inputs/valid', 'outputs', 'inputs/high_confidence', 'inputs/low_confidence']
    for folder in folders.copy():
        folders += _class_folders(folder)

    folders += ['inputs/models', 'videos', 'temp']
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
