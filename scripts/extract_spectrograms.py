"""
This script is used to compute spectrograms on the training dataset.
"""
import os
import numpy as np

import sys
sys.path.append('..')
from utils import compute_spectrogram, ProgressBar


FILES_LOCATION = '/home/vqxafa/Documents/data/audio'
SPEC_SAVE_LOCATION = '/home/vqxafa/Documents/data/spectrograms_22000_1024_512_10pad'


if __name__ == "__main__":
    files = os.listdir(FILES_LOCATION)
    progress_bar = ProgressBar(len(files), 30, 'Computing Spectrograms...')
    progress_bar.update(0)

    existing_spectrograms = set(os.listdir(SPEC_SAVE_LOCATION))

    for idx, f in enumerate(files):
        progress_bar.update(idx+1)
        # if idx == 69164:  # segmentation fault
        #     continue
        if '{}.npy'.format(f.split('.')[0]) in existing_spectrograms:
            continue
        try:
            y = compute_spectrogram('{}/{}'.format(FILES_LOCATION, f))
            np.save('{}/{}.npy'.format(SPEC_SAVE_LOCATION, f.split('.')[0]), y)
        except Exception as e:
            print(e, f)
