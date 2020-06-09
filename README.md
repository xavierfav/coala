# COALA: Co-Aligned Autoencoders for Learning Semantically Enriched Audio Representations

This is the repository for the method presented in the paper: "COALA: Co-Aligned Autoencoders for Learning Semantically Enriched Audio Representations" by X. Favory, [K. Drossos](https://kdrossos.net), [T. Virtanen](https://tutcris.tut.fi/portal/en/persons/tuomas-virtanen(210e58bb-c224-40a9-bf6c-5b786297e841).html), and X. Serra.


<p align="center">
  <img src="https://user-images.githubusercontent.com/10927428/84180915-9331a580-aa88-11ea-847d-10a5886f6052.png" width="450" />
</p>


## Install python dependencies
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results of the paper

### Training the embedding models


If you want to train the embeddings from scratch, you will need to download the dataset at [ZENODO LINK]() and place the hdf5 files in the `hdf5_ds/` directory.
Then you can launch the training of an embedding model by running for instance:
```
python train_dual_ae.py 'configs/dual_ae_c.json'
```
The config file may be edited for instance to select which device to use for training (`'cuda'` or `'cpu'`).


### Downtream classification tasks

If you want to re-compute the classification accuracies on the downstream tasks, you will need to:
1. download the three datasets:
   - [Urban Sound 8K](https://urbansounddataset.weebly.com/urbansound8k.html)
   - [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html) and its [fault-filtered splited version](https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split).
   - [NSynth](https://magenta.tensorflow.org/datasets/nsynth)
2. place their content into the directory `data/` as following:

```
data
└─── UrbanSound8K
│     └─── audio
│     └─── metadata
└─── GTZAN
│     └─── genres
│     └─── test_filtered.txt
│     └─── train_filtered.txt
└─── nsynth
     └─── nsynth-train
           └─── audio_selected
     └───  nsynth-test
```

Keeping existing sub-directories as they are for each dataset.
However, for NSynth, you will have to manually create the audio_selected/ folder and put there the files that are listed in the values of the dictionary stored in `json/nsynth_selected_sounds_per_class.json`. When nothing is specified,
  

3. compute the embeddings with the pre-trained (or re-trained) embedding models runing the `encode.py` script.
  This will store the embedding files into the `data/embedding/` directory.


## Use the pre-trained embedding models

You can use the embedding models on your own data.
You will need to create your own script, but the idea is simple.
Here is a simple example to extract embedding chunks given an audio file:

```python
from encode import return_loaded_model, extract_audio_embedding_chunks
from models_t1000 import AudioEncoder

model = return_loaded_model(AudioEncoder, 'saved_models/dual_ae_c/audio_encoder_epoch_200.pt')
embedding, _ = extract_audio_embedding_chunks(model, '<path/to/audio/file>')
```


---
```
   __         __            
 /"  "\     /"  "\                            _    
(  (\  )___(  /)  )                          | |    
 \               /             ___ ___   __ _| | __ _ 
 /               \            / __/ _ \ / _` | |/ _` |
/    () ___ ()    \          | (_| (_) | (_| | | (_| |
|      (   )      |           \___\___/ \__,_|_|\__,_|
 \      \_/      /           
   \...__!__.../              
```
