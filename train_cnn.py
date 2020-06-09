from pathlib import Path
from itertools import chain
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
import json

from data_loader import InMemoryDataset
from models_t1000 import *


LEARNING_RATE = 5e-3
EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL_EVERY = 10
DEVICE = 'cuda'

TRAIN_DATASET_FILE = 'hdf5_ds/spec_tags_top_1000'
VALIDATION_DATASET_FILE = 'hdf5_ds/spec_tags_top_1000_val'


def train():
    # Data loaders
    loader_params = {
        'batch_size': BATCH_SIZE, 
        'shuffle': True, 
        'num_workers': 1,
        'drop_last': True,
    }

    dataset_train = InMemoryDataset(TRAIN_DATASET_FILE)
    dataset_val = InMemoryDataset(VALIDATION_DATASET_FILE)
    train_loader = DataLoader(dataset_train, **loader_params)
    val_loader = DataLoader(dataset_val, **loader_params)

    device = torch.device(DEVICE)
    print(device)

    # folder for model checkpoints
    model_checkpoints_folder = Path('saved_models', 'cnn')
    if not model_checkpoints_folder.exists():
        model_checkpoints_folder.mkdir()

    # models
    cnn = CNN().to(device)

    # optimizers
    cnn_opt = optim.SGD(cnn.parameters(), lr=LEARNING_RATE)

    loss_function = nn.BCELoss()

    # Training loop
    for epoch in range(1, EPOCHS + 1):

        # Train
        cnn.train()

        train_loss = 0

        for batch_idx, (data, tags, _) in enumerate(train_loader):
            x = data.view(-1, 1, 96, 96).to(device)
            tags = tags.float().to(device)

            z, y = cnn(x)

            loss = loss_function(y, tags)
           
            cnn_opt.zero_grad()
            loss.backward()
            cnn_opt.step()

            train_loss += loss.item()

            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        print('====> Epoch: {} Loss: {:.6f}'.format(
            epoch, train_loss / len(train_loader.dataset) * BATCH_SIZE))

        if epoch%SAVE_MODEL_EVERY == 0:
            torch.save(cnn.state_dict(),
                str(Path(f'saved_models', 'cnn', f'audio_encoder_epoch_{epoch}.pt')))

        # Validation
        cnn.eval()

        val_loss = 0

        with torch.no_grad():
            for i, (data, _, sound_ids) in enumerate(val_loader):
                x = data.view(-1, 1, 96, 96).clamp(0).to(device)
                tags = tags.float().clamp(0).to(device)

                z, y = cnn(x)

                loss = loss_function(y, tags)

                val_loss += loss.item()
            
            print('====> Test set loss: {:.6f}'.format(
                val_loss / len(val_loader.dataset) * BATCH_SIZE))


if __name__ == "__main__":
    train()
