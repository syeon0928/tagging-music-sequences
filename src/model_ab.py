import torch.nn.functional as F
import torch
import torch.nn as nn
import torchaudio


# Based on Choi et al. 2016: Automatic Tagging ...
# https://doi.org/10.48550/arXiv.1606.00298
# https://github.com/minzwon/sota-music-tagging-models
class FCN5(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50):
        super(FCN5, self).__init__()

        # Transform signal to mel spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Layer 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d((2, 4))

        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d((2, 4))

        # Layer 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool2d((3, 5))

        # Layer 5
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        self.mp5 = nn.MaxPool2d((4, 4))

        # Dense
        self.dense = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spec transforms
        x = self.spec(x)
        x = self.to_db(x)
        # x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Apply each layer in sequence
        x = self.mp1(self.relu1(self.bn1(self.conv1(x))))
        x = self.mp2(self.relu2(self.bn2(self.conv2(x))))
        x = self.mp3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mp4(self.relu4(self.bn4(self.conv4(x))))
        x = self.mp5(self.relu5(self.bn5(self.conv5(x))))

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        # x = nn.Sigmoid()(x)

        return x
