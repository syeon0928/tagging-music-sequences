import torch
from torch import nn
import torch.nn.functional as F


# Define the CRNN model
class CRNN1(nn.Module):
    def __init__(self, num_classes=50):
        super(CRNN1, self).__init__()

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
        print(x.shape)
        # Apply 2D CNN layers
        x1 = self.elu1(self.bn1(self.conv1(x)))
        x1 = self.dropout1(x1)
        print("conv1 output shape:", x1.shape)

        x2 = self.elu2(self.bn2(self.conv2(x1)))
        x2 = self.dropout2(x2)
        print("conv2 output shape:", x2.shape)

        x3 = self.elu3(self.bn3(self.conv3(x2)))
        x3 = self.dropout3(x3)
        print("conv3 output shape:", x3.shape)

        x4 = self.elu4(self.bn4(self.conv4(x3)))
        x4 = self.dropout4(x4)
        print("conv4 output shape:", x4.shape)
    # ([2, 512, 96, 1819])        # Reshape for GRU
#         (N, L, H     )
#         x4 = x4.permute(0, 3, 2, 1)
        x4 = x4.permute(0, 3, 2, 1).contiguous().view(x4.size(0), -1, x4.size(1))
        # batch_size, channels, height, width = x4.shape
        print(x4.shape)
        # x4 = x4.reshape(width, channels, height)  # Reshape to [batch_size, width, channels * height]

        # Apply GRU layers
        x5, _ = self.gru1(x4)
        print(x5.shape)
        x6, _ = self.gru2(x5)
        print("gru2 output shape:", x6.shape)

        # Take the output from the last time step
        x6 = x6[:, -1, :]

        # Fully connected layer
        x7 = self.fc(x6)
        print("fc output shape:", x7.shape)

        return torch.sigmoid(x7)




class CRNN2(nn.Module):
    def __init__(self, num_classes=50):
        super(CRNN2, self).__init__()

        # Input block
        self.conv_block0 = nn.Sequential(
            nn.ZeroPad2d((0, 37)),
            nn.BatchNorm2d(1)
        )

        # Conv block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(0.1)
        )

        # Conv block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Dropout2d(0.1)
        )

        # Conv block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            nn.Dropout2d(0.1)
        )

        # Conv block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            nn.Dropout2d(0.1)
        )

        # Reshaping
        self.reshape = nn.Sequential(
            # Permute dimensions to (batch_size, channels, height, width)
        )

        # GRU block
        self.gru_block = nn.Sequential(
            nn.GRU(input_size=128, hidden_size=32, num_layers=2, batch_first=True, dropout=0.3),
            nn.Dropout(0.3)
        )

        # Output layer
        self.output_layer = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input block
        x = self.conv_block0(x)

        # Conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Reshaping
        x = x.permute(0, 3, 2, 1)  # Permute dimensions to (batch_size, width, channels, height)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape to (batch_size, width, channels * height)

        # GRU block
        x, _ = self.gru_block(x)

        # Output layer
        x = self.output_layer(x[:, -1, :])
        x = self.sigmoid(x)

        return x

# Instantiate the model

