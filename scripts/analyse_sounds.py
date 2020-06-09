"""
This script is used to extract low-level acoustic features on the validation set.
"""
import sys
sys.path.append('..')
import os
import json
import numpy as np
import librosa
import scipy
from tqdm import tqdm
import soundfile as sf

from utils import pad


# Directory where the files are
AUDIO_FILES_LOCATION = '/mnt/f/audio/'


def compute_spectrogram_and_log_mel_spectrogram(filename, sr=22000):
    try:
        audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    except:
        try:
            audio, o_sr = sf.read(filename)
            audio = librosa.core.resample(audio, o_sr, sr)
        except RuntimeError:
            return None, None
    try:
        x = pad(audio, sr)
    except ValueError:
        x = audio
    y_log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=sr, hop_length=512, n_fft=1024))
    y = np.abs(librosa.stft(y=x, hop_length=512, n_fft=1024))
    return y, y_log_mel
    

def return_spectrogram_max_nrg_frame(spectrogram, log_mel_spectrogram):
    frames_s = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    frames_s_log_meg = librosa.util.frame(np.asfortranarray(log_mel_spectrogram), frame_length=96, hop_length=12)
    idx_max_nrg = np.argmax(np.sum(np.sum(frames_s, axis=0), axis=0))
    return frames_s[:,:,idx_max_nrg], frames_s_log_meg[:,:,idx_max_nrg]


def compute_acoustic_features(filename, sr=22000):
    try:
        s, s_log_mel = compute_spectrogram_and_log_mel_spectrogram(filename, sr=sr)
        s, s_log_mel = return_spectrogram_max_nrg_frame(s, s_log_mel)

        mfcc = librosa.feature.mfcc(S=s_log_mel, sr=sr)
        chroma = librosa.feature.chroma_stft(S=s, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(S=s, sr=sr)
        spectral_bandwitdh = librosa.feature.spectral_bandwidth(S=s, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=s, sr=sr)

        return {
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_var': np.var(mfcc, axis=1).tolist(),
            'mfcc_skew': scipy.stats.skew(mfcc, axis=1).tolist(),
            'chroma_mean': np.mean(chroma, axis=1).tolist(),
            'chroma_var': np.var(chroma, axis=1).tolist(),
            'chroma_skew': scipy.stats.skew(chroma, axis=1).tolist(),
            'spectral_centroid': np.mean(spectral_centroid, axis=1).tolist(),
            'spectral_centroid_var': np.var(spectral_centroid, axis=1).tolist(),
            'spectral_centroid_skew': scipy.stats.skew(spectral_centroid, axis=1).tolist(),
            'spectral_bandwitdh': np.mean(spectral_bandwitdh, axis=1).tolist(),
            'spectral_bandwitdh_var': np.var(spectral_bandwitdh, axis=1).tolist(),
            'spectral_bandwitdh_skew': scipy.stats.skew(spectral_bandwitdh, axis=1).tolist(),
        }
    except:
        return None


if __name__ == "__main__":
    sound_ids = set(json.load(open('../json/sound_ids_validation.json', 'rb')))
    files = os.listdir(AUDIO_FILES_LOCATION)
    files_to_analyze = [AUDIO_FILES_LOCATION+f for f in files if int(f.split('.')[0]) in sound_ids]
    del files
    print(f'{len(files_to_analyze)} audio files to analyze...')

    sound_analysis = {}
    for f in tqdm(files_to_analyze):
        analysis = compute_acoustic_features(f)
        if analysis:
            sound_analysis[int(f.split('.')[0].split('/')[-1])] = analysis

    json.dump(sound_analysis, open('../json/sound_analysis_validation_set_librosa.json', 'w'))
