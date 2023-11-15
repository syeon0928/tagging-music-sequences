from torch.utils.data import Dataset, DataLoader
from src.audio_util import *
import torchaudio


class AudioDS(Dataset):
    def __init__(self, df, data_path, output='waveform'):
        self.df = df
        self.data_path = str(data_path)
        self.output = output
        self.class_columns = df.drop(columns=['filepath']).columns.to_list()

        # Convert each column in class_columns to boolean
        for col in self.class_columns:
            df[col] = df[col].astype('bool')

        # Define standardization instance variables
        self.duration = 30
        self.sample_rate = 16000
        self.channel = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Concatenated file path - Example: '../data/' + 'mtat/.......mp3'
        audio_file = self.data_path + self.df.loc[idx, 'filepath']

        # Retrieve labels
        label = self.df.loc[idx, self.class_columns].astype(bool).to_numpy()
        label = torch.from_numpy(label)

        # Load audio as tuple: (waveform, sample_rate)
        waveform, sample_rate = AudioUtil.open(audio_file)

        # Set sampling rate and audio length
        waveform, sample_rate = AudioUtil.resample(waveform, sample_rate, self.sample_rate)
        # waveform, sample_rate = AudioUtil.rechannel(waveform, sample_rate, self.channel)
        waveform, sample_rate = AudioUtil.fix_audio_length(waveform, sample_rate, self.duration)

        # Return Mel spectrogram and labels if specified, otherwise return waveform and labels
        if self.output == 'mel_spec':
            mel_spec_db = AudioUtil.mel_spectrogram_with_db(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None)
            return mel_spec_db, label
        else:
            waveform, sample_rate = waveform, sample_rate
            return waveform, label

    # decode labels based on class columns
    def decode_labels(self, encoded_labels):
        # Decodes the one-hot encoded labels back to their class names
        decoded_labels = []
        for i, label in enumerate(encoded_labels):
            if label:  # If the label is True (or 1)
                decoded_labels.append(self.class_columns[i])
        return decoded_labels

    # Get file path at a given index
    def get_filepath(self, idx):
        # Retrieve the file path for a given index
        return self.data_path + self.df.loc[idx, 'filepath']




