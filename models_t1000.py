"""
This file contains the pytorch model definitions for the dataset using 
the top 1000 select tags.
"""
import torch
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Sigmoid, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, LeakyReLU


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 3, 3)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = Sequential( 
            Conv2d(1, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            Flatten(),
        )

        self.fc_audio = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(0.25),
        )

    def forward(self, x):
        z = self.audio_encoder(x)
        z_d = self.fc_audio(z)
        return z, z_d


class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()

        self.audio_decoder = Sequential(
            UnFlatten(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(1),
            Sigmoid(),
        )

    def forward(self, z):
        return self.audio_decoder(z)


class TagEncoder(nn.Module):
    def __init__(self):
        super(TagEncoder, self).__init__()

        self.tag_encoder = Sequential(
            Linear(1000, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 1152),
            BatchNorm1d(1152),
            ReLU(),
            Dropout(.25),
        )

        self.fc_tag = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(.25),
        )

    def forward(self, tags):
        z = self.tag_encoder(tags)
        z_d = self.fc_tag(z)
        return z, z_d


class TagDecoder(nn.Module):
    def __init__(self):
        super(TagDecoder, self).__init__()

        self.tag_decoder = Sequential(
            Linear(1152, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1000),
            BatchNorm1d(1000),
            Sigmoid(),
        )

    def forward(self, z):
        return self.tag_decoder(z)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.encoder = Sequential( 
            Conv2d(1, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            Flatten(),
        )

        self.fc = Sequential(
                Linear(1152, 1152),
                ReLU(),
                Linear(1152, 1000),
                Sigmoid(),
            )

    def forward(self, x):
        z = self.encoder(x)
        y = self.fc(z)
        return z, y


# https://github.com/minzwon/sota-music-tagging-models/
class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class AudioEncoderRes(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self,
                n_channels=128,
                embedding_size=1152):
        super(AudioEncoderRes, self).__init__()

        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, embedding_size)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.3)
        self.dropout6 = nn.Dropout(0.3)
        self.dropout7 = nn.Dropout(0.3)
        self.dropout8 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc_audio = Sequential(
            Linear(embedding_size, embedding_size, bias=False),
            Dropout(0.25),
        )

    def forward(self, x):
        # CNN
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        x = self.layer5(x)
        x = self.dropout5(x)
        x = self.layer6(x)
        x = self.dropout6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dropout7(x)
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.dropout8(x)
        x = self.dense2(x)
        z = self.relu2(x)

        z_d = self.fc_audio(z)

        return z, z_d
