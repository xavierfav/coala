"""
This file regroups the procedures for training the neural networks.
A training uses a configuration json file (e.g. configs/dual_ae_c.json).
"""
from pathlib import Path
from itertools import chain
import torch
from torch.utils import data
from torch import nn, optim
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
import json

from data_loader import InMemoryDataset
from models_t1000 import *
from utils import kullback_leibler, contrastive_loss


class DualAETrainer():
    def __init__(self, params):
        self.params = params
        self.audio_encoder = None
        self.audio_decoder = None 
        self.tag_encoder = None
        self.tag_decoder = None
        self.train_dataset_file = params['train_dataset_file']
        self.validation_dataset_file = params['validation_dataset_file']
        self.audio_loss_weight = params['audio_loss_weight']
        self.tag_loss_weight = params['tag_loss_weight']
        self.contrastive_loss_weight = params['contrastive_loss_weight']
        self.contrastive_temperature = params['contrastive_temperature']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        self.experiment_name = params['experiment_name']
        self.id2tag_file = params['id2tag_file']
        self.log_interval = params['log_interval']
        self.save_model_every = params['save_model_every']

    def init_models(self):
        # self.audio_encoder = AudioEncoder()
        self.audio_encoder = AudioEncoderRes()
        self.audio_decoder = AudioDecoder()
        self.tag_encoder = TagEncoder()
        self.tag_decoder = TagDecoder()

    def load_model_checkpoints(self):
        saved_models_folder = Path('saved_models', self.experiment_name)
        try:
            last_epoch = max([int(f.stem.split('epoch_')[-1]) for f in saved_models_folder.iterdir()])
            self.audio_encoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'audio_encoder_epoch_{last_epoch}.pt'))))
            self.audio_decoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'audio_decoder_epoch_{last_epoch}.pt'))))
            self.tag_encoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'tag_encoder_epoch_{last_epoch}.pt'))))
            self.tag_decoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'tag_decoder_epoch_{last_epoch}.pt'))))
            print(f'Model checkpoints from epoch {last_epoch} loaded...')
        except ValueError:
            last_epoch = 0
            print('No model loaded, training from scratch...')

        self.iteration_idx = last_epoch * int(self.length_val_dataset / self.batch_size)
        self.last_epoch = last_epoch

    def train(self):
        """ Train the dual Auto Encoder   
        
        """
        # Data loaders
        loader_params = {
            'batch_size': self.batch_size, 
            'shuffle': True, 
            'num_workers': 1,
            'drop_last': True,
        }

        dataset_train = InMemoryDataset(self.train_dataset_file)
        dataset_val = InMemoryDataset(self.validation_dataset_file)

        self.train_loader = data.DataLoader(dataset_train, **loader_params)
        self.val_loader = data.DataLoader(dataset_val, **loader_params)
        self.length_train_dataset = len(self.train_loader.dataset)
        self.length_val_dataset = len(self.val_loader.dataset)

        # mapping id2tags
        self.id2tag = json.load(open(self.id2tag_file, 'rb'))

        # folder for model checkpoints
        model_checkpoints_folder = Path('saved_models', self.experiment_name)
        if not model_checkpoints_folder.exists():
            model_checkpoints_folder.mkdir()

        # models
        self.init_models()
        self.load_model_checkpoints()

        self.audio_encoder.to(self.device)
        self.audio_decoder.to(self.device)
        self.tag_encoder.to(self.device)
        self.tag_decoder.to(self.device)

        # optimizers
        self.audio_dae_opt = optim.SGD(chain(self.audio_encoder.parameters(), self.audio_decoder.parameters()), lr=self.learning_rate)
        self.tag_dae_opt = optim.SGD(chain(self.tag_encoder.parameters(), self.tag_decoder.parameters()), lr=self.learning_rate)

        # loss for tag autoencoder
        self.tag_recon_loss_function = torch.nn.BCELoss()

        # tensorboard
        with SummaryWriter(log_dir=str(Path('runs', self.experiment_name)), max_queue=100) as self.tb:

            # Training loop
            for epoch in range(self.last_epoch+1, self.epochs + 1):
                self.train_one_epoch_dual_AE(epoch)
                self.val_dual_AE(epoch)

    def train_one_epoch_dual_AE(self, epoch):
        """ Train one epoch

        """
        self.audio_encoder.train()
        self.audio_decoder.train()
        self.tag_encoder.train()
        self.tag_decoder.train()

        # losses
        train_audio_recon_loss = 0
        train_tags_recon_loss = 0
        train_loss = 0
        train_pairwise_loss = 0

        for batch_idx, (data, tags, sound_ids) in enumerate(self.train_loader):
            self.iteration_idx += 1

            x = data.view(-1, 1, 96, 96).to(self.device)
            tags = tags.float().to(self.device)

            # encode
            z_audio, z_d_audio = self.audio_encoder(x)
            z_tags, z_d_tags = self.tag_encoder(tags)

            # audio reconstruction
            x_recon = self.audio_decoder(z_audio)
            audio_recon_loss = kullback_leibler(x_recon, x)

            # tags reconstruction
            tags_recon = self.tag_decoder(z_tags)
            tags_recon_loss = self.tag_recon_loss_function(tags_recon, tags)

            # contrastive loss
            pairwise_loss = contrastive_loss(z_d_audio, z_d_tags, self.contrastive_temperature)

            # total loss
            loss = audio_recon_loss + tags_recon_loss + pairwise_loss

            # Optimize models
            self.audio_dae_opt.zero_grad()
            self.tag_dae_opt.zero_grad()
            audio_recon_loss.mul(self.audio_loss_weight).backward(retain_graph=True)
            tags_recon_loss.mul(self.tag_loss_weight).backward(retain_graph=True)
            if self.contrastive_loss_weight:
                pairwise_loss.mul(self.contrastive_loss_weight).backward()
            self.audio_dae_opt.step()
            self.tag_dae_opt.step()

            train_audio_recon_loss += audio_recon_loss.item()
            train_tags_recon_loss += tags_recon_loss.item()
            train_loss += loss.item()
            train_pairwise_loss += pairwise_loss.item()

            # write to tensorboard
            if False:
                self.tb.add_scalar("iter/audio_recon_loss", audio_recon_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/tag_recon_loss", tags_recon_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/contrastive_pairwise_loss", pairwise_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/total_loss", loss.item(), self.iteration_idx)

            # logs per batch
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} Audio Recon: {:.4f}, '
                      'Tags Recon: {:.4f},  Pairwise: {:.4f})'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item(),
                        audio_recon_loss.item(),
                        tags_recon_loss.item(),
                        pairwise_loss.item()
                    )
                )

        # epoch logs
        train_loss = train_loss / self.length_train_dataset * self.batch_size
        train_audio_recon_loss = train_audio_recon_loss / self.length_train_dataset * self.batch_size
        train_tags_recon_loss = train_tags_recon_loss / self.length_train_dataset * self.batch_size
        train_pairwise_loss = train_pairwise_loss / self.length_train_dataset * self.batch_size
        
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
        print('recon loss audio: {:.4f}'.format(train_audio_recon_loss))
        print('recon loss tags: {:.4f}'.format(train_tags_recon_loss))
        print('pairwise loss: {:.8f}'.format(train_pairwise_loss))
        print('\n')

        # tensorboard
        self.tb.add_scalar("audio_recon_loss/train", train_audio_recon_loss, epoch)
        self.tb.add_scalar("tag_recon_loss/train", train_tags_recon_loss, epoch)
        self.tb.add_scalar("contrastive_pairwise_loss/train", train_pairwise_loss, epoch)
        self.tb.add_scalar("total_loss/train", train_loss, epoch)

        if epoch%self.save_model_every == 0:
            torch.save(self.audio_encoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'audio_encoder_epoch_{epoch}.pt')))
            torch.save(self.audio_decoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'audio_decoder_epoch_{epoch}.pt')))
            torch.save(self.tag_encoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'tag_encoder_epoch_{epoch}.pt')))
            torch.save(self.tag_decoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'tag_decoder_epoch_{epoch}.pt')))

    def val_dual_AE(self, epoch):
        """ Validation dual autoencoder

        """
        self.audio_encoder.eval()
        self.audio_decoder.eval()
        self.tag_encoder.eval()
        self.tag_decoder.eval()

        val_audio_recon_loss = 0
        val_tags_recon_loss = 0
        val_loss = 0
        val_pairwise_loss = 0

        with torch.no_grad():
            for i, (data, tags, sound_ids) in enumerate(self.val_loader):
                # replace negative values with 0 using clamp. Negative values can appear in the 
                # validation set because the minmax scaler is learned on the training data only.
                x = data.view(-1, 1, 96, 96).clamp(0).to(self.device)
                tags = tags.float().clamp(0).to(self.device)

                # encode
                z_audio, z_d_audio = self.audio_encoder(x)
                z_tags, z_d_tags = self.tag_encoder(tags)
                
                # audio
                x_recon = self.audio_decoder(z_audio)
                audio_recon_loss = kullback_leibler(x_recon, x)

                # tags
                tags_recon = self.tag_decoder(z_tags)
                tags_recon_loss = self.tag_recon_loss_function(tags_recon, tags)

                # pairwise correspondence loss
                pairwise_loss = contrastive_loss(z_d_audio, z_d_tags, self.contrastive_temperature)

                loss = audio_recon_loss + tags_recon_loss + pairwise_loss

                val_audio_recon_loss += audio_recon_loss.item()
                val_tags_recon_loss += tags_recon_loss.item()
                val_loss += loss.item()
                val_pairwise_loss += pairwise_loss.item()

                # display some examples
                if i == 0:
                    n = min(data.size(0), 8)

                    # write files with original and reconstructed spectrograms
                    comparison = torch.cat([x.flip(2)[:n],
                                        x_recon.view(self.batch_size, 1, 96, 96).flip(2)[:n]])
                    save_image(comparison.cpu(),
                               f'reconstructions/reconstruction_{self.experiment_name}_{epoch}.png', nrow=n)

                    # print the corresponding reconstructed tags if id2tag is passed
                    if self.id2tag:
                        for idx in range(n):
                            print('\n',sound_ids.cpu()[idx].tolist()[0], 
                                sorted(zip(tags_recon.cpu()[idx].tolist(), 
                                        [self.id2tag[str(k)] for k in range(len(tags))]), reverse=True)[:6])
                        print('\n')

        val_loss = val_loss / self.length_val_dataset * self.batch_size
        val_audio_recon_loss = val_audio_recon_loss / self.length_val_dataset * self.batch_size
        val_tags_recon_loss = val_tags_recon_loss / self.length_val_dataset * self.batch_size
        val_pairwise_loss = val_pairwise_loss / self.length_val_dataset * self.batch_size

        print('====> Val average loss: {:.4f}'.format(val_loss))
        print('recon loss audio: {:.4f}'.format(val_audio_recon_loss))
        print('recon loss tags: {:.4f}'.format(val_tags_recon_loss))
        print('pairwise loss: {:.4f}'.format(val_pairwise_loss))
        print('\n\n')

        # tensorboard
        self.tb.add_scalar("audio_recon_loss/val", val_audio_recon_loss, epoch)
        self.tb.add_scalar("tag_recon_loss/val", val_tags_recon_loss, epoch)
        self.tb.add_scalar("contrastive_pairwise_loss/val", val_pairwise_loss, epoch)
        self.tb.add_scalar("total_loss/val", val_loss, epoch)
        