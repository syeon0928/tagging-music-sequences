import torch.nn.functional as F
import torch
import torch.nn as nn


# Architecture similar to this paper:
# Evaluation of CNN-based Automatic Music Tagging Models (arXiv:2006.00751)
class FCN4(nn.Module):
    def __init__(self, num_classes=50):
        super(FCN4, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 384, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(384, 768, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(768, 2048, kernel_size=(3, 3), padding=1)

        # Max-pooling layers
        self.pool1 = nn.MaxPool2d((2, 4))
        self.pool2 = nn.MaxPool2d((4, 5))
        self.pool3 = nn.MaxPool2d((3, 8))
        self.pool4 = nn.MaxPool2d((4, 8))

        self.final_conv = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Apply final convolution to get to the desired label shape (batch_size, num_classes, 1, 1)
        x = self.final_conv(x)

        # Flattening the output for the label shape: torch.Size([batch_size, num_classes])
        x = torch.flatten(x, 1)

        # Apply sigmoid activation function
        x = torch.sigmoid(x)

        return x


class FCN4BatchNormDropout(nn.Module):
    def __init__(self, num_classes=50):
        super(FCN4BatchNormDropout, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 384, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(384)
        self.conv3 = nn.Conv2d(384, 768, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = nn.Conv2d(768, 2048, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(2048)

        # Max-pooling layers
        self.pool1 = nn.MaxPool2d((2, 4))
        self.pool2 = nn.MaxPool2d((4, 5))
        self.pool3 = nn.MaxPool2d((3, 8))
        self.pool4 = nn.MaxPool2d((4, 8))

        # Dropout layers
        self.dropout = nn.Dropout(0.5)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.dropout(self.pool1(F.relu(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(self.pool2(F.relu(x)))
        x = self.bn3(self.conv3(x))
        x = self.dropout(self.pool3(F.relu(x)))
        x = self.bn4(self.conv4(x))
        x = self.dropout(self.pool4(F.relu(x)))

        # Apply final convolution to get to the desired label shape (batch_size, num_classes, 1, 1)
        x = self.final_conv(x)

        # Flattening the output for the label shape: torch.Size([batch_size, num_classes])
        x = torch.flatten(x, 1)

        # Apply sigmoid activation function
        x = torch.sigmoid(x)

        return x


# Based on: Evaluation of CNN-based Automatic Music Tagging Models (arXiv:2006.00751)
class FCN5(nn.Module):
    def __init__(self, num_classes=50):
        super(FCN5, self).__init__()

        # Define convolutional layers with the specified number of feature maps
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=1)

        # Define max-pooling layers with specified strides
        self.pool1 = nn.MaxPool2d((2, 4))
        self.pool2 = nn.MaxPool2d((2, 4))
        self.pool3 = nn.MaxPool2d((2, 4))
        self.pool4 = nn.MaxPool2d((3, 5))
        self.pool5 = nn.MaxPool2d((4, 4))

        # Final convolutional layer to adjust the output to num_classes dimensions
        self.final_conv = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        # Apply final convolution to get to the desired label shape (batch_size, 50, 1, 1)
        x = self.final_conv(x)

        # Flattening the output for the label shape: torch.Size([batch_size, 50])
        x = torch.flatten(x, 1)

        return x


class FCN5BatchNormDropout(nn.Module):
    def __init__(self, num_classes=50):
        super(FCN5BatchNormDropout, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(2048)

        # Max-pooling layers
        self.pool1 = nn.MaxPool2d((2, 4))
        self.pool2 = nn.MaxPool2d((2, 4))
        self.pool3 = nn.MaxPool2d((2, 4))
        self.pool4 = nn.MaxPool2d((3, 5))
        self.pool5 = nn.MaxPool2d((4, 4))

        # Dropout layers
        self.dropout = nn.Dropout(0.5)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.dropout(self.pool1(F.relu(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(self.pool2(F.relu(x)))
        x = self.bn3(self.conv3(x))
        x = self.dropout(self.pool3(F.relu(x)))
        x = self.bn4(self.conv4(x))
        x = self.dropout(self.pool4(F.relu(x)))
        x = self.bn5(self.conv5(x))
        x = self.dropout(self.pool5(F.relu(x)))

        # Apply final convolution to get to the desired label shape (batch_size, num_classes, 1, 1)
        x = self.final_conv(x)

        # Flattening the output for the label shape: torch.Size([batch_size, num_classes])
        x = torch.flatten(x, 1)

        return x
