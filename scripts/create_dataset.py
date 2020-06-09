"""
This script is used to create the HDF5 training and validation dataset files.
"""
import sys
sys.path.append('..')
import os
import json
import pickle
import h5py
import numpy as np
from collections import Counter
import pandas as pd
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LsiModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from utils import ProgressBar, return_spectrogram_max_nrg_frame, return_spectrogram_3_max_nrg_frames
NUM_BANDS = 96
NUM_FRAMES = 96

STRATEGY = 'top_1000'

# write to
SAVE_DATASET_LOCATION = '../hdf5_ds'
DATASET_NAME_TAGS = 'spec_tags_{}'.format(STRATEGY)
ID2TOKEN_NAME = '../json/id2token_{}.json'.format(STRATEGY)
SCALER_NAME = '../scaler_{}.pkl'.format(STRATEGY)

# read from
SOUND_TAGS = '../tags/tags_ppc_{}.csv'.format(STRATEGY)
SPECTROGRAM_LOCATION = '/mnt/f/data/spectrograms_22000_1024_512_10pad'  # SET FOLDER HERE!


if __name__ == "__main__":
    # load sound tags and create label idx vectors
    sound_tags_data = pd.read_csv(SOUND_TAGS, error_bad_lines=False)
    num_sounds = sound_tags_data.shape[0]
    sound_tags = [[t for t in sound_tags_data.iloc[idx].tolist()[1:] if isinstance(t, str)] for idx in range(num_sounds)]
    sound_ids_tags = list(sound_tags_data['id'])

    # remove sounds that does not have spectrograms
    progress_bar = ProgressBar(len(sound_ids_tags), 30, '1st pass...')
    progress_bar.update(0)
    idx_to_remove = []

    for idx, sound_id in enumerate(sound_ids_tags):
        progress_bar.update(idx+1)
        try:
            # spec
            x = np.load('{}/{}.npy'.format(SPECTROGRAM_LOCATION, sound_id))
            if not x.any():
                idx_to_remove.append(idx)
                continue

        except Exception as e:
            idx_to_remove.append(idx)

    idx_to_remove = set(idx_to_remove)

    # remove
    sound_ids = [j for i, j in enumerate(sound_ids_tags) if i not in idx_to_remove]
    sound_tags = [j for i, j in enumerate(sound_tags) if i not in idx_to_remove]
    num_sounds = len(sound_ids)
    print('\n Removed {} sounds\n'.format(len(idx_to_remove)))

    # extract tag vector
    dictionary = corpora.Dictionary(sound_tags)
    num_tags = len(dictionary)
    label_idx_vectors = [[0]*num_tags for _ in range(num_sounds)]
    bow_corpus = [[t[0] for t in dictionary.doc2bow(tags)] for tags in sound_tags]
    for idx, bow in enumerate(bow_corpus):
        for label_idx in bow:
            label_idx_vectors[idx][label_idx] = 1

    # split train val
    msss = MultilabelStratifiedShuffleSplit(test_size=0.1, random_state=0, n_splits=1)
    r = list(msss.split(sound_ids, label_idx_vectors))
    train_idx = r[0][0]
    val_idx = r[0][1]

    num_training_instances = len(train_idx)
    num_validation_instances = len(val_idx)

    print('Num training instances: {}'.format(num_training_instances))
    print('Num validation instances: {}'.format(num_validation_instances))

    train_ids = [sound_ids[idx] for idx in train_idx]
    train_labels = [label_idx_vectors[idx] for idx in train_idx]
    val_ids = [sound_ids[idx] for idx in val_idx]
    val_labels = [label_idx_vectors[idx] for idx in val_idx]

    # scaler
    scaler = MinMaxScaler()  # scale beetween 0 and 1
    progress_bar = ProgressBar(len(train_ids), 20, 'Learn scaler')
    progress_bar.update(0)
    for idx, sound_id in enumerate(train_ids):
        x = return_spectrogram_max_nrg_frame(np.load('{}/{}.npy'.format(SPECTROGRAM_LOCATION, sound_id)))
        for i in range(x_frames.shape[-1]):
            scaler.partial_fit(x_frames[...,i])
        progress_bar.update(idx+1)
    pickle.dump(scaler, open(SCALER_NAME, 'wb'))

    # save idx to token json
    id2token = {v: k for k, v in dictionary.token2id.items()}
    json.dump(id2token, open(ID2TOKEN_NAME, 'w'))

    # tag label dataset
    hdf5_file_tags = h5py.File('{}/{}'.format(SAVE_DATASET_LOCATION, DATASET_NAME_TAGS), mode='w')
    ds_group = hdf5_file_tags.create_group('dataset')
    ds_group.create_dataset("id", (num_training_instances, 1), dtype='int32')
    ds_group.create_dataset("data", (num_training_instances, NUM_FRAMES, NUM_BANDS), dtype='float32')
    ds_group.create_dataset("label", (num_training_instances, num_tags), dtype='int16')

    hdf5_file_tags_val = h5py.File('{}/{}_val'.format(SAVE_DATASET_LOCATION, DATASET_NAME_TAGS), mode='w')
    ds_group_val = hdf5_file_tags_val.create_group('dataset')
    ds_group_val.create_dataset("id", (num_validation_instances, 1), dtype='int32')
    ds_group_val.create_dataset("data", (num_validation_instances, NUM_FRAMES, NUM_BANDS), dtype='float32')
    ds_group_val.create_dataset("label", (num_validation_instances, num_tags), dtype='int16')

    progress_bar = ProgressBar(len(train_ids), 20, 'Dataset train tags')
    progress_bar.update(0)
    count_chunks = 0

    for idx, (fs_id, label) in enumerate(zip(train_ids, train_labels)):
        try:
            progress_bar.update(idx)
            x = return_spectrogram_max_nrg_frame(np.load('{}/{}.npy'.format(SPECTROGRAM_LOCATION, fs_id)))
            for i in range(x_frames.shape[-1]):
                x = scaler.transform(x_frames[...,i])
                if x.any():
                    ds_group["id"][count_chunks] = int(fs_id)
                    ds_group["data"][count_chunks] = x
                    ds_group["label"][count_chunks] = np.array(label)

                    count_chunks += 1

        except Exception as e:
            print(e)
            pass
    
    print('\n Train Tags Dataset finished, created {} training instances from {} audio files'.format(count_chunks, len(train_ids)))

    progress_bar = ProgressBar(len(val_ids), 20, 'Dataset val tags')
    progress_bar.update(0)
    count_chunks = 0

    for idx, (fs_id, label) in enumerate(zip(val_ids, val_labels)):
        try:
            progress_bar.update(idx)
            x = return_spectrogram_max_nrg_frame(np.load('{}/{}.npy'.format(SPECTROGRAM_LOCATION, fs_id)))
            for i in range(x_frames.shape[-1]):
                x = scaler.transform(x_frames[...,i])
                if x.any():
                    ds_group_val["id"][count_chunks] = int(fs_id)
                    ds_group_val["data"][count_chunks] = x
                    ds_group_val["label"][count_chunks] = np.array(label)

                    count_chunks += 1

        except Exception as e:
            print(e)
            pass 

    print('\n Val Tags Dataset finished, created {} training instances from {} audio files\n'.format(count_chunks, len(val_ids)))
    hdf5_file_tags.close()
    hdf5_file_tags_val.close()
