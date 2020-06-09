"""
This script is used to pre-process tags from Freesound content.
"""
import sys
sys.path.append('..')
import json
import pandas as pd
import gensim
from gensim import corpora, models
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.lancaster import LancasterStemmer
import inflect

from utils import ProgressBar


SOUND_TAGS_FILE = './tags/fs_sound_inf_10_sec_07_05_19.csv'
SPECTROGRAM_LOCATION = '/home/xavierfav/Documents/dev/vgg_w2v/spectrograms'
FS_IDS_TARGET_TASK_FILE = './json/fs_ids_target_task.json'
OUTPUT_FILE = 'tags_ppc_top_1000'
KEEP_N = 1000

DICTIONARY = set(words.words())
st = LancasterStemmer()
p = inflect.engine()
STOPWORDS = set(stopwords.words('english'))


def singular_form(term):
    out = p.singular_noun(term)
    if out:
        return out
    else:
        return term


def preprocess(tags):
    result = []
    try:
        tags = tags.split(' ')
        tags = [item.lower() for item in tags if item.isalpha()]
    except AttributeError:  # NaN
        return []
    for tag in tags:
        if tag not in STOPWORDS and len(tag) > 1:
            result.append(tag)
        else:
            pass
            # print(tag)
    result = [singular_form(tag) for tag in result]
            
    return list(set(result))


if __name__ == "__main__":
    data = pd.read_csv(SOUND_TAGS_FILE, error_bad_lines=False)

    # remove sounds from target task
    fs_ids_target_task = set(json.load(open(FS_IDS_TARGET_TASK_FILE, 'rb')))

    data = data[~data['fs_id'].isin(fs_ids_target_task)]
    data_tags = data[['tags']]
    data_tags['fs_id'] = data['fs_id']

    # preproc
    processed_tags = data_tags['tags'].map(preprocess)
    print(processed_tags[:10])

    # filter tags
    dictionary = corpora.Dictionary(processed_tags)
    # filter:     
    #   less than 1000 documents (absolute number) or
    #   more than 0.3 documents (fraction of total corpus size, not absolute number).
    #   after the above two steps, keep only the first 200 most frequent tokens.
    dictionary.filter_extremes(no_below=10, no_above=0.7, keep_n=KEEP_N)  # These params seem to give good and few tags
    corpus = [[dictionary[tag_id] for tag_id in dictionary.doc2idx(tags) if tag_id != -1] for tags in processed_tags]
    print('\nExample bag of tags: {}\n'.format(corpus[:10]))

    # most occuring terms
    corpus_bow = [dictionary.doc2bow(sent) for sent in processed_tags]
    vocab_tf={}
    for i in corpus_bow:
        for item, count in dict(i).items():
            if item in vocab_tf:
                vocab_tf[item] += count
            else:
                vocab_tf[item] = count
    ordered_terms = [(dictionary[i], c) for i, c in sorted(vocab_tf.items(), key=lambda x: x[1], reverse=True)]
    print('\nMost occurring tags: {}\n'.format(ordered_terms[:100]))
    print('\Less occurring tags: {}\n'.format(ordered_terms[-100:]))

    max_num_tags = max([len(t) for t in corpus])
    print('\nMaximum {} tags \n'.format(max_num_tags))

    # create output file with tags
    progress_bar = ProgressBar(len(processed_tags), 20, 'Saving processed tags...')
    progress_bar.update(0)
    count = 1
    count_sounds_with_tags = 0

    with open('tags/{}.csv'.format(OUTPUT_FILE), 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(['id'] + list(range(max_num_tags)))
        for fs_id, tags in zip(data_tags['fs_id'], corpus):
            count += 1
            progress_bar.update(count)
            if tags:
                count_sounds_with_tags += 1
                writer.writerow([fs_id] + tags)
    print('\nNumber of sounds: {}'.format(count_sounds_with_tags))
    print('\nNumber of sounds with more than 10 tags: {}'.format(len([1 for t in corpus if len(t)>=10])))
    print('\nNumber of sounds with more than 5 tags: {}'.format(len([1 for t in corpus if len(t)>=5])))
    print('\nNumber of sounds with less than 5 tags: {}'.format(len([1 for t in corpus if len(t)<5])))
    print('\nTotal number of tags: {}'.format(len(dictionary)))

    # save dictionary mapping
    json.dump(dictionary.id2token, open('tags/{}_mapping.json'.format(OUTPUT_FILE), 'w'))
