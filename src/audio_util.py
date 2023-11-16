import math, random
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
from torchaudio.transforms import TimeStretch, FrequencyMasking
import matplotlib.pyplot as plt

class AudioUtil:
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate

    @staticmethod
    def get_audio_channels(waveform, sample_rate):
        num_channels = waveform.shape[0]  # The shape is (num_channels, num_samples)
        return num_channels

    @staticmethod
    def get_audio_duration(waveform, sample_rate):
        num_samples = waveform.shape[1]
        duration_seconds = num_samples / sample_rate
        return duration_seconds

    @staticmethod
    def resample(audio, new_sample_rate):
        waveform, sample_rate = audio

        if sample_rate == new_sample_rate:
            # Nothing to do
            return waveform, sample_rate

        num_channels = waveform.shape[0]
        # Resample first channel
        resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[:1,:])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            resample_second_channel = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[1:,:])
            resampled_waveform = torch.cat([resampled_waveform, resample_second_channel])

        return resampled_waveform, new_sample_rate

    @staticmethod
    def rechannel(audio, new_channel):
        waveform, sample_rate = audio

        if waveform.shape[0] == new_channel:
            # Nothing to do
            return waveform, sample_rate

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            resampled_waveform = waveform[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resampled_waveform = torch.cat([waveform, waveform])

        return resampled_waveform, sample_rate

    @staticmethod
    def pad_or_trunc(audio, max_s):
        waveform, sample_rate = audio
        num_rows, waveform_len = waveform.shape
        max_len = int(sample_rate * max_s)

        # Truncate to given length
        if waveform_len > max_len:
            waveform = waveform[:, :max_len]

        # Pad if it's shorter than max length
        elif waveform_len < max_len:
            pad_end_len = max_len - waveform_len

            # Pad with 0s at the end
            pad_end = torch.zeros((num_rows, pad_end_len))

            waveform = torch.cat((waveform, pad_end), 1)

        return waveform, sample_rate

    @staticmethod
    def get_audio_transforms(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=64, top_db=80):
        """
        Create a transformation pipeline for audio data.

        Parameters:
        - sample_rate: The sample rate for the audio.
        - n_fft: The size of FFT to use.
        - hop_length: The hop length for FFT.
        - n_mels: The number of Mel bands.
        - top_db: The threshold for the amplitude to dB conversion.

        Returns:
        - A PyTorch Sequential transformation.
        """
        return torch.nn.Sequential(
            T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
            T.AmplitudeToDB(top_db=top_db)
        )


class AudioPlot:
    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", ax=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        if ax is None:
            _, ax = plt.subplots(num_channels, 1)
        ax.plot(time_axis, waveform[0], linewidth=1)
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_title(title)

    @staticmethod
    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(specgram.numpy(), origin="lower", aspect="auto", interpolation="nearest")


class AudioAugment:

    @staticmethod
    def pitch_shift(audio, n_steps=4):
        waveform, sample_rate = audio
        waveform = waveform.numpy()
        waveform_pitch_shifted = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
        waveform_pitch_shifted = torch.from_numpy(waveform_pitch_shifted)
        return waveform_pitch_shifted, sample_rate

    @staticmethod
    def time_stretch(audio, stretch_factor=1.0):
        waveform, sample_rate = audio
        waveform = waveform.numpy()
        waveform_stretched = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        waveform_stretched = torch.from_numpy(waveform_stretched)
        return waveform_stretched, sample_rate

    @staticmethod
    def apply_augmentation(audio, augmentation, params):
        waveform, sample_rate = audio
        if augmentation == 'pitch_shift':
            return AudioAugment.pitch_shift(audio, **params)
        elif augmentation == 'time_stretch':
            return AudioAugment.time_stretch(audio, **params)
        # ... other augmentations ...
        else:
            return waveform, sample_rate

    @staticmethod
    def random_augment(waveform, sample_rate, augmentations):
        chosen_augmentation = random.choice(augmentations)
        augmentation_name = chosen_augmentation['name']
        params = chosen_augmentation.get('params', {})
        audio = (waveform, sample_rate)
        return AudioAugment.apply_augmentation(audio, augmentation_name, params)

    @staticmethod
    def augment_and_store_audio(df, n, output_dir, augmentations, dry_run=False):
        new_rows = []

        for index, row in df.iterrows():
            filepath = row['filepath']
            full_path = os.path.join('../data', filepath)  # Full path for reading

            waveform, sample_rate = AudioUtil.open(full_path)

            for i in range(n):
                augmented_waveform, _ = AudioAugment.random_augment(waveform.clone(), sample_rate, augmentations)
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                augmented_filename = f'{base_filename}_aug_{i}.wav'

                # Relative path for saving and appending to DataFrame
                relative_save_path = os.path.join(output_dir, augmented_filename)

                # Full path for saving the file
                full_save_path = os.path.join('../data', relative_save_path)

                if not dry_run:
                    # Save as .wav
                    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
                    torchaudio.save(full_save_path, augmented_waveform, sample_rate)

                # Prepare new row data
                new_row = row.copy()
                new_row['filepath'] = relative_save_path  # Store the relative path
                new_rows.append(new_row)

        # Concatenate the new rows to the original DataFrame
        new_rows_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows_df], ignore_index=True)

        return df

    # try new augmentor class
    class AudioDataAugmentor:
        def __init__(self, data_path, output_dir, augmentations, n_augmentations=3):
            self.data_path = data_path
            self.output_dir = output_dir
            self.augmentations = augmentations
            self.n_augmentations = n_augmentations
            self.train_original = None
            self.df_aug = None

        def load_data(self):
            self.train_original = pd.read_csv(self.data_path, index_col=0).reset_index(drop=True)

        def perform_augmentations(self):
            new_rows = []

            for index, row in self.train_original.iterrows():
                filepath = row['filepath']
                full_path = os.path.join('../data', filepath)

                waveform, sample_rate = self.torchaudio.load(filepath)

                for i in range(self.n_augmentations):
                    augmented_waveform, _ = self.random_augment(waveform.clone(), sample_rate)
                    augmented_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_aug_{i}.wav"
                    relative_save_path = os.path.join(self.output_dir, augmented_filename)
                    full_save_path = os.path.join('../data', relative_save_path)

                    if not dry_run:
                        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
                        torchaudio.save(full_save_path, augmented_waveform, sample_rate)

                    new_row = row.copy()
                    new_row['filepath'] = relative_save_path
                    new_rows.append(new_row)

            new_rows_df = pd.DataFrame(new_rows)
            self.df_aug = pd.concat([self.train_original, new_rows_df], ignore_index=True)

        # Augmentation methods
        @staticmethod
        def pitch_shift(audio, n_steps=4):
            waveform, sample_rate = audio
            waveform_shifted = librosa.effects.pitch_shift(waveform.numpy(), sr=sample_rate, n_steps=n_steps)
            return torch.from_numpy(waveform_shifted), sample_rate

        @staticmethod
        def time_stretch(audio, stretch_factor=1.0):
            waveform, sample_rate = audio
            waveform_stretched = librosa.effects.time_stretch(waveform.numpy(), rate=stretch_factor)
            return torch.from_numpy(waveform_stretched), sample_rate

        def apply_augmentation(self, audio, augmentation, params):
            if augmentation == 'pitch_shift':
                return self.pitch_shift(audio, **params)
            elif augmentation == 'time_stretch':
                return self.time_stretch(audio, **params)
            else:
                return audio

        def random_augment(self, waveform, sample_rate):
            chosen_augmentation = random.choice(self.augmentations)
            return self.apply_augmentation((waveform, sample_rate), chosen_augmentation['name'],
                                           chosen_augmentation.get('params', {}))
