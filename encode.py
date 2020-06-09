"""
This script is used to compute neural network embeddings.
"""
import torch
import numpy as np
import sklearn
import pickle
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import librosa

from utils import compute_spectrogram
from models_t1000 import AudioEncoder, TagEncoder, CNN


scaler = pickle.load(open('../scaler_top_1000.pkl', 'rb'))
id2tag = json.load(open('../json/id2token_top_1000.json', 'rb'))
tag2id = {tag: id for id, tag in id2tag.items()}


def return_loaded_model(Model, checkpoint):
    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_audio_embedding(model, filename):
    with torch.no_grad():
        try:
            x = compute_spectrogram(filename)[:96, :96]
            x = scaler.transform(x)
            x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x), 0), 0).float()
            embedding, embedding_d = model(x)
            return embedding, embedding_d
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


def extract_audio_embedding_chunks(model, filename):
    with torch.no_grad():
        try:
            x = compute_spectrogram(filename)
            x_chunks = np.array([scaler.transform(chunk.T) for chunk in 
                    librosa.util.frame(np.asfortranarray(x), frame_length=96, hop_length=96, axis=-1).T])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            embedding_chunks, embedding_d_chunks = model(x_chunks)
            return embedding_chunks, embedding_d_chunks
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


def extract_tag_embedding(model, tag):
    with torch.no_grad():
        try:
            tag_vector = torch.tensor(np.zeros(1000)).view(1, 1000).float()
            tag_vector[0, int(tag2id[tag])] = 1
            embedding, _ = model(tag_vector)
            return embedding
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, tag)


if __name__ == "__main__":
    for MODEL_NAME in [
        'cnn/audio_encoder_epoch_20',
        'dual_ae_c/audio_encoder_epoch_200',
        'dual_e_c/audio_encoder_epoch_200',

    ]:
        MODEL_PATH = f'./saved_models/{MODEL_NAME}.pt'

        if 'cnn' in MODEL_NAME:
            model = return_loaded_model(CNN, MODEL_PATH)
        else:
            model = return_loaded_model(AudioEncoder, MODEL_PATH)

        # GTZAN
        p = Path('./data/GTZAN/genres')
        filenames_gtzan = p.glob('**/*.wav')

        # US8K
        p = Path('./data/UrbanSound8K/audio')
        filenames_us8k = p.glob('**/*.wav')

        # NSynth
        p = Path('./data/nsynth/nsynth-train/audio_selected')
        filenames_nsynth_train = p.glob('*.wav')
        p = Path('./data/nsynth/nsynth-test/audio')
        filenames_nsynth_test = p.glob('*.wav')


        dataset_files = [filenames_gtzan, filenames_us8k, filenames_nsynth_train, filenames_nsynth_test]
        dataset_names = ['gtzan', 'us8k', 'nsynth/train', 'nsynth/test']

        for filenames, ds_name in zip(dataset_files, dataset_names):

            print(f'\n {ds_name}  {MODEL_NAME}')

            for f in tqdm(filenames):
                try:
                    model_name = MODEL_NAME.split('/')[0] + '_' + MODEL_NAME.split('_epoch_')[-1]
                    folder = f'./data/embeddings/{ds_name}/embeddings_{model_name}'
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    embedding, embedding_d = extract_audio_embedding_chunks(model, str(f))
                    np.save(Path(folder, str(f.stem)+'.npy'), embedding)
                except Exception as e:
                    print(e)
            print('\n')
