"""
This script is used to plot the tag embeddings on validation set.
"""
import sys
sys.path.append('..')
import torch
import numpy as np
import os
import json
import sklearn
from matplotlib import pyplot as plt

from encode import return_loaded_model
from models_t1000 import TagEncoder


def plot_tag_embeddings():
    tag_encoder = return_loaded_model(TagEncoder, '../saved_models/dual_ae_c/tag_encoder_epoch_200.pt')
    id2tag = json.load(open('../json/id2token_top_1000.json', 'rb'))
    tag_embeddings = []

    for tag_idx, _ in id2tag.items():
        tag_idx = int(tag_idx)
        tag_vector = torch.tensor(np.zeros(1000)).view(1, 1000).float()
        tag_vector[0, tag_idx] = 1
        embedding, _ = tag_encoder(tag_vector)
        tag_embeddings.append(embedding.tolist())

    data = np.array(tag_embeddings).reshape(1000, 1152)
    tsne = sklearn.manifold.TSNE(n_components=2)
    tag_embeddings_tsne = tsne.fit_transform(data)    

    fig, ax = plt.subplots()
    for idx, (x, y) in enumerate(tag_embeddings_tsne):
        ax.scatter(x, y, alpha=0.8, c='red', edgecolors='none', s=5, marker="+")
        ax.annotate(id2tag[str(idx)], (x, y))

    plt.show()


if __name__ == "__main__":
    plot_tag_embeddings()
