#Models modified based on code: https://github.com/minzwon/self-attention-music-tagging/tree/master
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import math
import copy
import torch.nn.functional as F

# Gelu
def gelu(x):
    """Implementation of the gelu activation function.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# LayerNorm

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 attention_probs_dropout_prob=0.1,
                 type_vocab_size=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.classier = nn.Linear(768, 50)

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=True):
        batch_size = hidden_states.size()[0]
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        classier_head_output = self.classier(hidden_states)
        return classier_head_output.view(batch_size, -1)

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_ids + position_embeddings
        # embeddings = input_ids
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        emb_dim = config.hidden_size
        max_len = config.max_position_embeddings

        # Define position_enc as a learnable parameter
        self.position_enc = nn.Parameter(self.position_encoding_init(max_len, emb_dim), requires_grad=False)

    @staticmethod
    def position_encoding_init(n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        return torch.tensor(position_enc, dtype=torch.float32)  # Convert to PyTorch tensor

    def forward(self, word_seq):
        position_encoding = self.position_enc.unsqueeze(0).expand_as(word_seq)
        position_encoding = position_encoding.to(word_seq.device)
        word_pos_encoded = word_seq + position_encoding
        return word_pos_encoded


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FCN7FE(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=96,
                 attention_channels = 512
                 ):
        super(FCN7FE, self).__init__()

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

        # Additional 1x1 convolutional layers
        self.conv6 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        # Dense
        self.dense = nn.Linear(32, attention_channels)
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

        # Apply additional 1x1 convolutional layers
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)

        return x

class WaveCNN7FE(nn.Module):
    def __init__(self, attention_channels=512):
        super(WaveCNN7FE, self).__init__()

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

        # Fully connected layers
        self.fc1 = nn.Linear(512, attention_channels)  # Adjust based on attention_channels
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        # Configuration
        bert_config = BertConfig(vocab_size=config.attention_channels,
                            hidden_size=config.attention_channels,
                            num_hidden_layers=config.attention_layers,
                            num_attention_heads=config.attention_heads,
                            intermediate_size=config.attention_channels*4,
                            hidden_act="gelu",
                            hidden_dropout_prob=config.attention_dropout,
                            max_position_embeddings=config.attention_length,
                            attention_probs_dropout_prob=config.attention_dropout)

        # Embedding (Feature map + Positional encoding + Mask)
        #self.embedding = BertEmbeddings(bert_config)
        self.embedding = PositionalEncoding(bert_config)

        # Bert encoder
        self.encoder = BertEncoder(bert_config)

        # Bert pooler
        self.pooler = BertPooler(bert_config)

    def forward(self, x):
        x = self.embedding(x)
        encoded_layers = self.encoder(x)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


class FrontEndBackEndModel(nn.Module):
    def __init__(self, front_end, back_end):
        super(FrontEndBackEndModel, self).__init__()
        self.front_end = front_end
        self.back_end = back_end

    def load_pretrained_parameters(self, front_end_pretrained_path='models/FCN7_best_l2_20231201-2215.pth', freeze_layers=5):
        # Load pre-trained parameters for the front-end model
        pretrained_state_dict = torch.load(front_end_pretrained_path)

        # Load pre-trained parameters, skipping unnecessary keys
        self.front_end.load_state_dict(
            {k: v for k, v in pretrained_state_dict.items() if k in self.front_end.state_dict()})

        # Freeze layers in the front-end model up to the specified layer
        for i, param in enumerate(self.front_end.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

    def forward(self, x):
        # Front-end
        front_end_output = self.front_end(x)

        # Back-end
        back_end_output = self.back_end(front_end_output)

        return back_end_output

class Config(object):
    def __init__(self,
                attention_channels=512,
                attention_layers=2,
                attention_heads=8,
                attention_length=257,
                attention_dropout=0.1):
        self.attention_channels = attention_channels
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_length = attention_length
        self.attention_dropout = attention_dropout



