import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src import audio_util
from src.audio_augmentations import PitchShiftAugmentation, TimeStretchAugmentation


class AudioDS(Dataset):
    def __init__(
            self,
            annotations_file,
            data_dir,
            target_sample_rate=16000,
            target_length=29.1,
            transformations=None,
            augmentations=None,
    ):
        self.annotations_file = annotations_file
        self.data_dir = data_dir
        self.sample_rate = target_sample_rate
        self.target_length = target_length
        self.transformations = transformations
        self.augmentations = augmentations

        # Load annotations using pandas
        self.annotations_file = pd.read_csv(os.path.join(data_dir, annotations_file), index_col=0).reset_index(
            drop=True)

        # Convert each column in class_columns to float
        self.class_columns = self.annotations_file.drop(columns=["filepath"]).columns.to_list()
        for col in self.class_columns:
            self.annotations_file[col] = self.annotations_file[col].astype("float")

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        # Concatenated file path - Example: '../data/' + 'mtat/.......mp3'
        audio_file = os.path.join(self.data_dir, self.annotations_file.loc[idx, "filepath"])

        # Retrieve labels
        label = (self.annotations_file.loc[idx, self.class_columns].astype(float).to_numpy())
        label = torch.from_numpy(label)

        # Load audio as unpacked tuple: waveform, sample_rate
        signal, sample_rate = audio_util.open(audio_file)

        # Set sampling rate and audio length
        signal, sample_rate = audio_util.resample(signal, sample_rate, self.sample_rate)
        signal, sample_rate = audio_util.pad_or_trunc(signal, sample_rate, self.target_length)

        # transform to mel spec
        if self.transformations:
            signal = self.transformations(signal)

            # augment mel spec (we only do augmentation on mel spec)
            if self.augmentations:
                signal = self.augmentations(signal)

        return signal, label, audio_file

    # Get file path at a given index
    def get_filepath(self, idx):
        # Retrieve the file path for a given index
        return os.path.join(self.data_dir, self.annotations_file.loc[idx, "filepath"])

    def decode_labels(self, encoded_labels):
        # Decodes the one-hot encoded labels back to their class names
        decoded_labels = []
        for i, label in enumerate(encoded_labels):
            if label:  # If the label is True (or 1)
                decoded_labels.append(self.class_columns[i])
        return decoded_labels


def get_dataloader(
        annotations_file,
        data_dir,
        batch_size,
        shuffle,
        num_workers,
        sample_rate,
        target_length,
        transformations=None,
        augmentations=None,
):

    dataset = AudioDS(
        annotations_file=annotations_file,
        data_dir=data_dir,
        target_sample_rate=sample_rate,
        target_length=target_length,
        transformations=transformations,
        augmentations=augmentations,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
