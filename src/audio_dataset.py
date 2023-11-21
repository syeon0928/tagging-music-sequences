import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src import audio_util


class AudioDS(Dataset):
    def __init__(self,
                 annotations_file,
                 data_dir,
                 target_sample_rate=16000,
                 target_length=30,
                 transformation=None,
                 ):

        self.annotations_file = annotations_file
        self.data_dir = data_dir
        self.sample_rate = target_sample_rate
        self.target_length = target_length
        self.transformation = transformation

        # Load annotations using pandas
        self.annotations_file = pd.read_csv(os.path.join(data_dir, annotations_file), index_col=0).reset_index(
            drop=True)

        # Convert each column in class_columns to float
        self.class_columns = self.annotations_file.drop(columns=['filepath']).columns.to_list()
        for col in self.class_columns:
            self.annotations_file[col] = self.annotations_file[col].astype('float')

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        # Concatenated file path - Example: '../data/' + 'mtat/.......mp3'
        audio_file = os.path.join(self.data_dir, self.annotations_file.loc[idx, 'filepath'])

        # Retrieve labels
        label = self.annotations_file.loc[idx, self.class_columns].astype(float).to_numpy()
        label = torch.from_numpy(label)

        # Load audio as tuple: (waveform, sample_rate)
        audio = audio_util.open(audio_file)

        # Set sampling rate and audio length
        audio = audio_util.resample(audio, self.sample_rate)
        audio = audio_util.pad_or_trunc(audio, self.target_length)

        # Apply transformation if it's provided
        if self.transformation:
            signal, _ = audio
            transformed_signal = self.transformation(signal)
            return transformed_signal, label
        else:
            signal, _ = audio
            return signal, label

    # Get file path at a given index
    def get_filepath(self, idx):
        # Retrieve the file path for a given index
        return os.path.join(self.data_dir, self.annotations_file.loc[idx, 'filepath'])

    def decode_labels(self, encoded_labels):
        # Decodes the one-hot encoded labels back to their class names
        decoded_labels = []
        for i, label in enumerate(encoded_labels):
            if label:  # If the label is True (or 1)
                decoded_labels.append(self.class_columns[i])
        return decoded_labels


def get_dataloader(annotations_file, data_dir, batch_size, shuffle, sample_rate, target_length, transform_params=None):
    # Apply transformations if transform_params is provided
    transformation = audio_util.get_audio_transforms(**transform_params) if transform_params else None

    dataset = AudioDS(
        annotations_file=annotations_file,
        data_dir=data_dir,
        target_sample_rate=sample_rate,
        target_length=target_length,
        transformation=transformation
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
