"""
This script is used to plot the audio and tag based embeddings for the clips 
from the validation set.
"""
import sys
import torch
import numpy as np
import sklearn
import pickle
import os
import json
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('../')
from data_loader import HDF5Dataset
from encode import extract_tag_embedding, extract_audio_embedding, return_loaded_model
from models_t1000 import AudioEncoder, TagEncoder


N = 800

loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
dataset_val = HDF5Dataset('../hdf5_ds/spec_tags_top_1000_val')
test_loader = data.DataLoader(dataset_val, **loader_params)


def extract_audio_tag_embeddings():
    sound_id2idx = {id:idx for idx, id in 
                    enumerate(list(dataset_val.h_file['dataset']['id'][:,0]))}

    audio_encoder = return_loaded_model(AudioEncoder, '../saved_models/dual_ae_c/audio_encoder_epoch_200.pt')
    tag_encoder = return_loaded_model(TagEncoder, '../saved_models/dual_ae_c/tag_encoder_epoch_200.pt')

    audio_embeddings = []
    tag_embeddings = []
    sound_ids = []

    for idx, (data, tags, sound_id) in enumerate(test_loader):
        sound_ids += sound_id.tolist()
        x = data.view(-1, 1, 96, 96).clamp(0)
        tags = tags.float().clamp(0)

        # encode
        z_audio, z_d_audio = audio_encoder(x)
        z_tags, z_d_tags = tag_encoder(tags)

        audio_embeddings.append(z_audio.tolist())
        tag_embeddings.append(z_tags.tolist())

        if idx == N:
            break

    size_embedding = z_tags.shape[1]
    audio_embeddings = np.array(audio_embeddings).reshape(len(audio_embeddings), size_embedding)[:N, :]
    tag_embeddings = np.array(tag_embeddings).reshape(len(tag_embeddings), size_embedding)[:N, :]

    data = np.concatenate((audio_embeddings, tag_embeddings), 0)
    tsne = sklearn.manifold.TSNE(n_components=2)
    data = tsne.fit_transform(data)
    audio_embeddings_tsne = data[:N, :]
    tag_embeddings_tsne = data[N:, :]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for idx, (x, y) in enumerate(audio_embeddings_tsne):
        ax.scatter(x, y, alpha=0.8, c='red', edgecolors='none', s=8)
        # ax.annotate(sound_ids[idx][0], (x, y))

    for idx, (x, y) in enumerate(tag_embeddings_tsne):
        ax.scatter(x, y, alpha=0.8, c='blue', edgecolors='none', s=8)
        # ax.annotate(sound_ids[idx][0], (x, y))

    for (x0,y0), (x1,y1) in zip(audio_embeddings_tsne, tag_embeddings_tsne):
        ax.plot((x0,x1), (y0,y1), linewidth=0.2, color='black', alpha=0.6)

    plt.title('Visualisation of the aligned learnt representations (TSNE)')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='audio',
               markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='tags',
               markerfacecolor='blue', markersize=5),
    ]  
    plt.legend(handles=legend_elements, loc=2)
    plt.show()
    

if __name__ == "__main__":
    extract_audio_tag_embeddings()
