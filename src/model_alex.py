import torch
from torch import nn
import torch.nn.functional as F


# Architecture similar to this paper:
# Evaluation of CNN-based Automatic Music Tagging Models (arXiv:2006.00751)
class FullyConvNet(nn.Module):
    def __init__(self):
        super(FullyConvNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        # Define max-pooling layers with different strides
        self.pool1 = nn.MaxPool2d((2, 4))
        self.pool2 = nn.MaxPool2d((4, 5))
        self.pool3 = nn.MaxPool2d((3, 8))
        self.pool4 = nn.MaxPool2d((4, 8))

        # The final convolutional layer to adjust the output to the desired number of classes
        # Note: The number of output features needs to be tuned based on the final output size after the last pooling layer
        self.final_conv = nn.Conv2d(128, 50, kernel_size=(1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Apply final convolution to get to the desired label shape (batch_size, 50, 1, 1)
        x = self.final_conv(x)

        # Flattening the output for the label shape: torch.Size([batch_size, 50])
        x = torch.flatten(x, 1)

        return x