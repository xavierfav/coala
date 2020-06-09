"""
This script is used to perform the correlation analysis beetwen learned embeddings and 
low-level acoustic features.
It uses code from SVCCA (https://github.com/google/svcca), repo that needs to be
cloned in the parent directory of COALA.
"""
import sys
import torch
import numpy as np
import sklearn
import pickle
import os
import json
from torch.utils import data as dt
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import mplcursors

sys.path.append('..')
sys.path.append('../svcca/')
import cca_core

from data_loader import HDF5Dataset
from encode import return_loaded_model
from models_t1000 import AudioEncoder, TagEncoder, CNN


MODEL_CHECKPOINTS = {
    'ae_c': '../saved_models/dual_ae_c/audio_encoder_epoch_200.pt',
    'e_c': '../saved_models/dual_e_c/audio_encoder_epoch_200.pt',
    'cnn': '../saved_models/cnn/audio_encoder_epoch_20.pt',
}

loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
dataset_val = HDF5Dataset('hdf5_ds/spec_tags_top_1000_val')


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def extract_audio_embedding(model_checkpoint):
    """ Returns sound_ids and their respective audio embeddings """
    test_loader = dt.DataLoader(dataset_val, **loader_params)
    
    if 'cnn' in model_checkpoint:
        audio_encoder = return_loaded_model(CNN, model_checkpoint)
    else:
        audio_encoder = return_loaded_model(AudioEncoder, model_checkpoint)
    
    audio_embeddings = []
    sound_ids = []

    for idx, (data, tags, sound_id) in enumerate(test_loader):
        sound_ids += sound_id.tolist()[0]
        x = data.view(-1, 1, 96, 96).clamp(0)

        # encode
        z_audio, z_d_audio = audio_encoder(x)
        audio_embeddings.append(z_audio.tolist())

    return sound_ids, np.array(audio_embeddings).squeeze()

    
def compute_cca(embeddings, descriptors):
    results = cca_core.get_cca_similarity(embeddings, descriptors, verbose=False, epsilon=1e-20)
    print(np.mean(results["cca_coef1"]))


if __name__ == "__main__":
    sound_analysis_dict = json.load(open('../json/sound_analysis_validation_set_librosa.json', 'r'))

    for model_name, model_checkpoint in MODEL_CHECKPOINTS.items():
        print(f'\nModel: {model_name}')
        sound_ids, audio_embeddings = extract_audio_embedding(model_checkpoint)
        sound_idx_to_remove = []
        sound_descriptors = []

        for idx, sound_id in enumerate(sound_ids):
            try:
                if sound_analysis_dict[str(sound_id)]:
                    sound_descriptors.append(sound_analysis_dict[str(sound_id)])
                else:
                    sound_idx_to_remove.append(idx)
            except:
                sound_idx_to_remove.append(idx)
            
        embeddings = np.delete(audio_embeddings, sound_idx_to_remove, axis=0)

        for feature_name in [
            'mfcc_mean',
            'mfcc_var',
            'mfcc_skew',
            'chroma_mean',
            'chroma_var',
            'chroma_skew',
            'spectral_centroid',
            'spectral_centroid_var',
            'spectral_centroid_skew',
            'spectral_bandwitdh',
            'spectral_bandwitdh_var',
            'spectral_bandwitdh_skew',
            'spectral_rolloff',
            'spectral_rolloff_var',
            'spectral_rolloff_skew',
        ]:
            descriptors =  np.array([s[feature_name] for s in sound_descriptors])
            print(feature_name)
            if len(descriptors.shape) < 2:
                descriptors = np.expand_dims(descriptors, axis=1)
            compute_cca(embeddings.T, descriptors.T)
