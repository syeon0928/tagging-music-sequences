import torch
from torch import nn
import torch.nn.functional as F


# Define the CRNN model
class CRNN(nn.Module):
    def __init__(self, num_classes=50):
        super(CRNN, self).__init__()

        # 2D CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout2d(0.1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout2d(0.1)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout2d(0.1)

        # GRU layers
        self.gru1 = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1)

        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply 2D CNN layers
        x = self.elu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.elu2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.elu3(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = self.elu4(self.bn4(self.conv4(x)))
        x = self.dropout4(x)

        # Reshape for GRU
        # Use view instead of permute to maintain the correct dimensions
        x = x.view(x.size(0), x.size(1), -1)

        # Apply GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Take the output from the last time step
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)

        return torch.sigmoid(x)
