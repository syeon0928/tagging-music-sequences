import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):

    def __init__(self, in_dim, heads):
        super(SelfAttentionLayer, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads

        assert (
            self.head_dim * heads == in_dim
        ), "Embedding dimension needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, in_dim)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class WaveCNN7WithSelfAttention(nn.Module):

    def __init__(self, num_classes=50, attention_heads=2):
        super(WaveCNN7WithSelfAttention, self).__init__()

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        out_channels = 128
        for i in range(7):
            if i == 3:  # 4th layer
                out_channels = 256
            if i == 6:  # Last layer
                out_channels = 512

            self.conv_blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1))
            self.conv_blocks.append(nn.BatchNorm1d(out_channels))
            self.conv_blocks.append(nn.ReLU())
            self.conv_blocks.append(nn.MaxPool1d(kernel_size=3, stride=3))
            in_channels = out_channels

        # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Self-attention layers
        self.self_attention1 = SelfAttentionLayer(in_dim=512, heads=attention_heads)
        self.self_attention2 = SelfAttentionLayer(in_dim=512, heads=attention_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Self-attention layers
        x = self.self_attention1(x, x, x, mask=None)
        x = self.self_attention2(x, x, x, mask=None)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class FCN5WithSelfAttention(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 num_classes=50,
                 attention_heads=2
                 ):
        super(FCN5WithSelfAttention, self).__init__()

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
        self.attention1 = SelfAttentionLayer(in_dim=64, heads=attention_heads)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d((2, 4))
        self.attention2 = SelfAttentionLayer(in_dim=128, heads=attention_heads)

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
        x = self.spec_bn(x)

        # Apply each layer in sequence
        x = self.mp1(self.relu1(self.bn1(self.conv1(x))))
        x = self.attention1(x, x, x, mask=None)
        x = self.mp2(self.relu2(self.bn2(self.conv2(x))))
        x = self.attention2(x, x, x, mask=None)
        x = self.mp3(self.relu3(self.bn3(self.conv3(x))))
        x = self.mp4(self.relu4(self.bn4(self.conv4(x))))
        x = self.mp5(self.relu5(self.bn5(self.conv5(x))))

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)

        return x