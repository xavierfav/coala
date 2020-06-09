"""
This script is used to compute mffc features for target task datasets.
Warning: Need manual editing for switching datasets
"""
import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path


FILES_LOCATION = '../data/UrbanSound8K/audio'
FILES_LOCATION = '../data/GTZAN/genres'
SAVE_LOCATION = '../data/embeddings/gtzan/mfcc'
SAVE_LOCATION = '../data/embeddings/nsynth/test/mfcc'


def compute_mfcc(filename, sr=22000):
    # zero pad and compute log mel spec
    try:
        audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    except:
        audio, o_sr = sf.read(filename)
        audio = librosa.core.resample(audio, o_sr, sr)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc, width=5, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=5, mode='nearest')
    
    feature = np.concatenate((np.mean(mfcc, axis=1), np.var(mfcc, axis=1),
                              np.mean(mfcc_delta, axis=1), np.var(mfcc_delta, axis=1),
                              np.mean(mfcc_delta2, axis=1), np.var(mfcc_delta2, axis=1)))

    return feature


if __name__ == "__main__":
    # p = Path(FILES_LOCATION)
    # filenames = p.glob('**/*.wav')
    # # filenames = p.glob('*')
    
    p = Path('../data/nsynth/nsynth-test/audio')
    filenames = p.glob('*.wav')

    for f in tqdm(filenames):
        try:
            y = compute_mfcc(str(f))
            np.save(Path(SAVE_LOCATION, str(f.stem)+'.npy'), y)
        except RuntimeError as e:
            print(e, f)
