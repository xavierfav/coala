"""
This script is used to evaluate the embeddings on the target tasks.
You'll need to download the datasets, compute the embeddings and set the path accordingly.
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import chain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import json


US8K_MEDATADA_FILE = './data/UrbanSound8K/metadata/UrbanSound8K.csv'
GTZAN_TRAIN_FILE = './data/GTZAN/train_filtered.txt'
GTZAN_TEST_FILE = './data/GTZAN/test_filtered.txt'


EMBEDDING_FOLDERS_US8K = {
    'mfcc': './data/embeddings/us8k/mfcc/',
    'cnn': './data/embeddings/us8k/embeddings_cnn',
    'e_c': './data/embeddings/us8k/embeddings_dual_e_c/',
    'ae_c': './data/embeddings/us8k/embeddings_dual_ae_c/',
}

EMBEDDING_FOLDERS_GTZAN = {
    'mfcc': './data/embeddings/gtzan/mfcc/',
    'cnn': './data/embeddings/gtzan/embeddings_cnn/',
    'e_c_200': './data/embeddings/gtzan/embeddings_dual_e_c/',
    'ae_c_200': './data/embeddings/gtzan/embeddings_dual_ae_c/',
}


EMBEDDING_FOLDERS_NSYNTH = {
    'mfcc': ('./data/embeddings/nsynth/train/mfcc/',
             './data/embeddings/nsynth/test/mfcc/'),
    'cnn': ('./data/embeddings/nsynth/train/embeddings_cnn/',
            './data/embeddings/nsynth/test/embeddings_cnn/'),
    'ae_c': ('./data/embeddings/nsynth/train/embeddings_dual_ae_c/',
            './data/embeddings/nsynth/test/embeddings_dual_ae_c/'),
    'e_c': ('./data/embeddings/nsynth/train/embeddings_dual_e_c/',
            './data/embeddings/nsynth/test/embeddings_dual_e_c/'),
}


GTZAN_CLASS_MAPPING = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}

NSYNTH_CLASS_MAPPING = {
    'brass': 0, 
    'guitar': 1, 
    'string': 2, 
    'vocal': 3, 
    'flute': 4, 
    'keyboard': 5, 
    'reed': 6, 
    'organ': 7, 
    'mallet': 8, 
    'bass': 9
}


# --------------- US8K ---------------
def create_folds(embedding_folder):
    # slice_file_name	fsID	start	end	    salience	fold	classID	    class
    data = pd.read_csv(US8K_MEDATADA_FILE, error_bad_lines=False).values.tolist()
    folds = [defaultdict(list) for _ in range(10)]
    for d in data:
        try:
            fold_idx = d[5]-1
            class_idx = d[6]
            file_name = d[0]
            folds[fold_idx]['X'].append(np.load(Path(embedding_folder, f'{file_name.split(".")[0]}.npy')))
            folds[fold_idx]['y'].append(class_idx)
        except:
            pass
    return folds

def return_other_fold_indexes(test_fold_idx):
    return [i for i in range(10) if i != test_fold_idx]

def eval_US8K(embedding_folder):
    folds = create_folds(embedding_folder)

    scores = []

    for fold_idx, test_fold in enumerate(folds):
        other_fold_indexes = return_other_fold_indexes(fold_idx)
        X = np.array(list(chain(*[folds[idx]['X'] for idx in other_fold_indexes]))).squeeze()
        y = np.array(list(chain(*[folds[idx]['y'] for idx in other_fold_indexes])))
        X_test = np.array(test_fold['X']).squeeze()
        y_test = np.array(test_fold['y'])

        if len(X_test.shape) > 2:
            X = X.mean(axis=1)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        clf = MLPClassifier(hidden_layer_sizes=(256,))
        clf.fit(X, y)

        if len(X_test.shape) > 2:
            X_test = X_test.mean(axis=1)
        X_test = scaler.transform(X_test)
        scores.append(clf.score(X_test, y_test))

    print(f'\nScores: {scores}, mean: {np.mean(scores)}\n')

    return np.mean(scores_svm)


# --------------- GTZAN ---------------
def create_dataset_gtzan(embedding_folder):
    train_files = pd.read_csv(GTZAN_TRAIN_FILE, error_bad_lines=False).values.tolist()
    test_files = pd.read_csv(GTZAN_TEST_FILE, error_bad_lines=False).values.tolist()

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for f_name in train_files:
        f_name = Path(f_name[0]).stem
        f = Path(embedding_folder, f'{f_name}.npy')
        label_idx = GTZAN_CLASS_MAPPING[f.stem.split('.')[0]]
        X_train.append(np.load(f))
        y_train.append(label_idx)
    
    for f_name in test_files:
        f_name = Path(f_name[0]).stem
        f = Path(embedding_folder, f'{f_name}.npy')
        label_idx = GTZAN_CLASS_MAPPING[f.stem.split('.')[0]]
        X_test.append(np.load(f))
        y_test.append(label_idx)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

def eval_gtzan_fault_filtered(embedding_folder):
    X_train, X_test, y_train, y_test = create_dataset_gtzan(embedding_folder)

    print("aggregate and scale...")
    if len(X_train.shape) > 2:
        X_train = X_train.mean(axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    print("train...")
    clf = MLPClassifier(hidden_layer_sizes=(256,))
    clf.fit(X_train, y_train)

    print("eval...")
    if len(X_test.shape) > 2:
        X_test = X_test.mean(axis=1)
    X_test = scaler.transform(X_test)
    score = clf.score(X_test, y_test)

    print(f'\MLP score: {score}\n')
    return score


# --------------- NSYNTH ---------------
def create_dataset_nsynth(embedding_folder_train, embedding_folder_test):
    print("loading train data...")
    p = Path(embedding_folder_train)
    X_train = []
    y_train = []
    for f in tqdm(p.iterdir()):
        try:
            if '_d' not in f.stem:
                label_idx = NSYNTH_CLASS_MAPPING[f.stem.split('_')[0]]
                X_train.append(np.load(f))
                y_train.append(label_idx)
        except:
            pass

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("loading test data...")
    p = Path(embedding_folder_test)
    X_test = []
    y_test = []
    for f in tqdm(p.iterdir()):
        try:
            if '_d' not in f.stem:
                label_idx = NSYNTH_CLASS_MAPPING[f.stem.split('_')[0]]
                X_test.append(np.load(f))
                y_test.append(label_idx)
        except:
            pass

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test
    
def eval_nsynth(embedding_folder_train, embedding_folder_test):
    X_train, X_test, y_train, y_test = create_dataset_nsynth(embedding_folder_train, embedding_folder_test)

    print("aggregate and scale...")
    if len(X_train.shape) > 2:
        X_train = X_train.mean(axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    print("train...")
    clf = MLPClassifier(hidden_layer_sizes=(256,))
    clf.fit(X_train, y_train)

    print("eval...")
    if len(X_test.shape) > 2:
        X_test = X_test.mean(axis=1)
    X_test = scaler.transform(X_test)
    score = clf.score(X_test, y_test)

    print(f'\MLP score: {score}\n')
    return score


if __name__ == "__main__":
    performances = {
        'NSynth': defaultdict(list),
        'US8K': defaultdict(list),
        'GTZAN': defaultdict(list),
    }

    for run_idx in range(10):

        # NSYNTH
        print('--------------- NSYNTH ---------------')
        for embedding_name, (embedding_folder, embedding_folder_test) in EMBEDDING_FOLDERS_NSYNTH.items():
            print(f'\nEmbedding: {embedding_name}')
            score = eval_nsynth(embedding_folder, embedding_folder_test)
            performances['NSynth'][embedding_name].append(score)
        print('\n\n')

        # US8K
        print('--------------- US8K ---------------')
        for embedding_name, embedding_folder in EMBEDDING_FOLDERS_US8K.items():
            print(f'\nEmbedding: {embedding_name}')
            score = eval_US8K(embedding_folder)
            performances['US8K'][embedding_name].append(score)
        print('\n\n')

        # GTZAN
        print('--------------- GTZAN ---------------')
        for embedding_name, embedding_folder in EMBEDDING_FOLDERS_GTZAN.items():
            print(f'\nEmbedding: {embedding_name}')
            score = eval_gtzan_fault_filtered(embedding_folder)
            performances['GTZAN'][embedding_name].append(score)
        print('\n\n')

    json.dump(performances, open('results/performances.json', 'w'))
