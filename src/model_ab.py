import torch.nn.functional as F
import torch
import torch.nn as nn
import torchaudio

from .modules import Conv_1d, Conv_2d, Conv_V, Conv_H

class FCN3(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50
                 ):
        super(FCN3, self).__init__()

        # Transform signal to mel spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d((2, 4))

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d((2, 4))

        # Dense
        self.dense = nn.Linear(64 * 12 * 28, num_classes) # Adjust the input size as needed
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        x = self.mp1(self.relu1(self.bn1(self.conv1(x))))
        x = self.mp2(self.relu2(self.bn2(self.conv2(x))))
        x = self.mp3(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)

        return x


class FCN4(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50
                 ):
        super(FCN4, self).__init__()

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
        self.mp2 = nn.MaxPool2d((4, 5))

        # Layer 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d((3, 8))

        # Layer 4
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool2d((4, 8))

        # Dense
        self.dense = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spec transforms
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # Apply each layer in sequence
        x = self.mp1(self.relu1(self.bn1(self.conv1(x))))
        x = self.mp2(self.relu2(self.bn2(self.conv2(x))))
        x = self.mp3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mp4(self.relu4(self.bn4(self.conv4(x))))

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)

        return x


# Based on Choi et al. 2016: Automatic Tagging ...
# https://doi.org/10.48550/arXiv.1606.00298
# https://github.com/minzwon/sota-music-tagging-models
class FCN5(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50
                 ):
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

        return x

class FCN7(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50
                 ):
        super(FCN7, self).__init__()

        # Transform signal to mel spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
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

        # Additional 1x1 convolutional layers
        self.conv6 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        # Dense
        self.dense = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spec transforms
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # Apply each layer in sequence
        x = self.mp1(self.relu1(self.bn1(self.conv1(x))))
        x = self.mp2(self.relu2(self.bn2(self.conv2(x))))
        x = self.mp3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mp4(self.relu4(self.bn4(self.conv4(x))))
        x = self.mp5(self.relu5(self.bn5(self.conv5(x))))

        # Apply additional 1x1 convolutional layers
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)

        return x

# Pons et al. 2017
# End-to-end learning for music audio tagging at scale.
# This is the updated implementation of the original paper. Referred to the Musicnn code.
# https://github.com/jordipons/musicnn
# https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
class MusicCNN(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50
                 ):
        super(MusicCNN, self).__init__()

        # Transform signal to mel spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # frontend
        # horizontal layers
        m1 = Conv_V(1, 204, (int(0.7*96), 7))
        m2 = Conv_V(1, 204, (int(0.4*96), 7))
        # vertical layers
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # backend
        backend_channel = 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # dense
        dense_channel = 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, num_classes)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # frontend
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


class FCN7_Transfer(nn.Module):
    def __init__(self, num_classes_new_task=10, pre_trained_model_path='models/FCN7_best_l2_20231201-2215.pth'):
        super(FCN7_Transfer, self).__init__()

        # Initialize the original FCN7 model
        self.original_fcn7 = FCN7(num_classes=50)

        # Load the pre-trained weights
        self.load_pretrained_weights(pre_trained_model_path)

        # Freeze the parameters of the original model
        for param in self.original_fcn7.parameters():
            param.requires_grad = False

        # Replace the last dense layer with new layers for the new task
        # Assume the new task has 'num_classes_new_task' classes
        self.new_layers = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes_new_task)
        )

    def load_pretrained_weights(self, path):
        checkpoint = torch.load(path)
        self.original_fcn7.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        # Pass input through the original model

        # Spec transforms
        x = self.original_fcn7.spec(x)
        x = self.original_fcn7.to_db(x)
        x = self.original_fcn7.spec_bn(x)

        # Apply each layer in sequence
        x = self.original_fcn7.mp1(self.original_fcn7.relu1(self.original_fcn7.bn1(self.original_fcn7.conv1(x))))
        x = self.original_fcn7.mp2(self.original_fcn7.relu2(self.original_fcn7.bn2(self.original_fcn7.conv2(x))))
        x = self.original_fcn7.mp3(self.original_fcn7.relu3(self.original_fcn7.bn3(self.original_fcn7.conv3(x))))
        x = self.original_fcn7.mp4(self.original_fcn7.relu4(self.original_fcn7.bn4(self.original_fcn7.conv4(x))))
        x = self.original_fcn7.mp5(self.original_fcn7.relu5(self.original_fcn7.bn5(self.original_fcn7.conv5(x))))

        # Apply additional 1x1 convolutional layers
        x = self.original_fcn7.relu6(self.original_fcn7.bn6(self.original_fcn7.conv6(x)))
        x = self.original_fcn7.relu7(self.original_fcn7.bn7(self.original_fcn7.conv7(x)))
        
        #Flatten the output to [batch_size, 32]
        x = x.view(x.size(0), -1)

        # Pass through new layers
        x = self.new_layers(x)

        return x
