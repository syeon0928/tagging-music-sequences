# Set path variables
import os
import sys
from pathlib import Path
cwd = os.getcwd()
project_dir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(project_dir)
data_path = os.path.join(project_dir, 'data/')
from src.audio_dataset import AudioDS
from torch.utils.data import DataLoader, Subset
from trainer import Trainer
import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class WaveCNN9(nn.Module):
    def __init__(self, num_classes=50):
        super(WaveCNN9, self).__init__()

        # Strided convolution to reduce dimensionality
        self.strided_conv = nn.Conv1d(1, 128, kernel_size=3, stride=3)
        self.bn0 = nn.BatchNorm1d(128)

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 128
        out_channels = 128
        for i in range(8):

            if i == 3:  # 5th layer
                out_channels = 256
            if i == 7:  # Last layer
                out_channels = 512

            self.conv_blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1))
            self.conv_blocks.append(nn.BatchNorm1d(out_channels))
            self.conv_blocks.append(nn.ReLU())
            self.conv_blocks.append(nn.MaxPool1d(kernel_size=3, stride=3))
            in_channels = out_channels

            # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        # Adjust the input size of the first FC layer based on the output of the last convolutional block
        self.fc1 = nn.Linear(512, 256)  # Adjust 128 based on the output channels of the last conv block
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial strided convolution
        x = F.relu(self.bn0(self.strided_conv(x)))

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class WaveCNN7(nn.Module):
    def __init__(self, num_classes=50):
        super(WaveCNN7, self).__init__()

        # Strided convolution to reduce dimensionality
        self.strided_conv = nn.Conv1d(1, 128, kernel_size=3, stride=3)
        self.bn0 = nn.BatchNorm1d(128)

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 128
        out_channels = 128
        for i in range(6):

            if i == 2:  # 4th layer
                out_channels = 256
            if i == 5:  # Last layer
                out_channels = 512

            self.conv_blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1))
            self.conv_blocks.append(nn.BatchNorm1d(out_channels))
            self.conv_blocks.append(nn.ReLU())
            self.conv_blocks.append(nn.MaxPool1d(kernel_size=3, stride=3))
            in_channels = out_channels

            # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        # Adjust the input size of the first FC layer based on the output of the last convolutional block
        self.fc1 = nn.Linear(out_channels, 256)  # Adjust 128 based on the output channels of the last conv block
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial strided convolution
        x = F.relu(self.bn0(self.strided_conv(x)))

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class WaveCNN5(nn.Module):
    def __init__(self, num_classes=50):
        super(WaveCNN5, self).__init__()

        # Strided convolution to reduce dimensionality
        self.strided_conv = nn.Conv1d(1, 128, kernel_size=3, stride=3)
        self.bn0 = nn.BatchNorm1d(128)

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 128
        out_channels = 128
        for i in range(4):

            if i == 1:  # 3rd layer
                out_channels = 256
            if i == 3:  # Last layer 5th
                out_channels = 512

            self.conv_blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1))
            self.conv_blocks.append(nn.BatchNorm1d(out_channels))
            self.conv_blocks.append(nn.ReLU())
            self.conv_blocks.append(nn.MaxPool1d(kernel_size=3, stride=3))
            in_channels = out_channels

            # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        # Adjust the input size of the first FC layer based on the output of the last convolutional block
        self.fc1 = nn.Linear(out_channels, 256)  # Adjust 128 based on the output channels of the last conv block
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial strided convolution
        x = F.relu(self.bn0(self.strided_conv(x)))

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x




if __name__=='__main__':

    # setup cuda device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'Using {device}')

    # Define global parameters across all classes
    SAMPLE_RATE = 16000
    DURATION_IN_SEC = 29.1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # data path
    cwd = Path.cwd()
    DATA_DIR = cwd.parent / 'data'

    # Load label annotation csv
    train_annotations = 'mtat_train_label.csv'
    val_annotations = 'mtat_val_label.csv'
    test_annotations = 'mtat_test_label.csv'

    # Load data
    train_data = AudioDS(annotations_file=train_annotations,
                         data_dir=DATA_DIR,
                         target_sample_rate=SAMPLE_RATE,
                         target_length=DURATION_IN_SEC,
                         transformation=None)

    val_data = AudioDS(annotations_file=val_annotations,
                       data_dir=DATA_DIR,
                       target_sample_rate=SAMPLE_RATE,
                       target_length=DURATION_IN_SEC,
                       transformation=None)

    test_data = AudioDS(annotations_file=val_annotations,
                        data_dir=DATA_DIR,
                        target_sample_rate=SAMPLE_RATE,
                        target_length=DURATION_IN_SEC,
                        transformation=None)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = next(iter(train_dataloader))

    ### CNN 9
    # # check the summary
    print('Start Training WaveCNN 7 Layers')
    wavecnn9 = WaveCNN9(num_classes=50)
    input_size = (train_features.size()[1:])
    model_summary = summary(wavecnn9.to(device), input_size) if device == 'cuda' else summary(wavecnn9, input_size)
    print(model_summary)
    
    # Train
    # Instantiate trainer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(wavecnn9.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(wavecnn9, train_dataloader, val_dataloader, criterion, optimizer, device)
    trainer.train(epochs=EPOCHS)

    ### CNN 7
    # check the summary
    print('Start Training WaveCNN 7 Layers')
    wavecnn7 = WaveCNN7(num_classes=50)
    input_size = (train_features.size()[1:])
    model_summary = summary(wavecnn7.to(device), input_size) if device == 'cuda' else summary(wavecnn7, input_size)
    print(model_summary)

    # Train
    # Instantiate trainer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(wavecnn7.parameters(), lr=LEARNING_RATE)
    trainer_7 = Trainer(wavecnn7, train_dataloader, val_dataloader, criterion, optimizer, device)

    trainer_7.train(epochs=EPOCHS)
    trainer_7.save_model('../models/waveform_cnn7.pth')
    print()

    ### CNN 5
    # check the summary
    print('Start Training WaveCNN 5 Layers')
    wavecnn5 = WaveCNN5(num_classes=50)
    input_size = (train_features.size()[1:])
    model_summary = summary(wavecnn5.to(device), input_size) if device == 'cuda' else summary(wavecnn5, input_size)
    print(model_summary)

    # Train
    # Instantiate trainer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(wavecnn5.parameters(), lr=LEARNING_RATE)
    traine_5 = Trainer(wavecnn5, train_dataloader, val_dataloader, criterion, optimizer, device)

    traine_5.train(epochs=EPOCHS)
    traine_5.save_model('../models/waveform_cnn5.pth')

