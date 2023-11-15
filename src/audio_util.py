import math, random
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torchaudio import transforms
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

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(waveform, sample_rate, new_channel):

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

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(waveform, sample_rate, new_sample_rate):

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

    # ----------------------------
    # Pad (or truncate) the waveform to a fixed length 'max_s' in seconds
    # ----------------------------
    @staticmethod
    def fix_audio_length(waveform, sample_rate, max_s):
        num_rows, waveform_len = waveform.shape
        max_len = int(sample_rate * max_s)

        # Truncate to given length
        if waveform_len > max_len:
            waveform = waveform[:, :max_len]

        # Pad if it's shorter than max length
        elif waveform_len < max_len:
            pad_begin_len = random.randint(0, max_len - waveform_len)
            pad_end_len = max_len - waveform_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            waveform = torch.cat((pad_begin, waveform, pad_end), 1)

        return waveform, sample_rate

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def mel_spectrogram_with_db(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

        # Convert to decibels
        spectrogram = transforms.AmplitudeToDB(top_db=top_db)(spectrogram)
        return spectrogram


class AudioPlot:
    # Plot waveform
    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        plt.figure(figsize=(14, 5))

        librosa.display.waveshow(waveform, sr=sample_rate)
        plt.title(title)

    @staticmethod
    def plot_spectrogram(spec, sample_rate, title="Spectrogram"):
        if isinstance(spec, torch.Tensor):
            # Convert PyTorch tensor to NumPy array
            spec = spec.cpu().detach().numpy()

            # Handle dimensions: spec should be 2D for plotting
        if spec.ndim > 2:
            # Average across the first dimension if it represents channels
            spec = np.mean(spec, axis=0)

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.title(title)
        plt.colorbar(format="%+2.f dB")


class AudioAugment:

    @staticmethod
    def pitch_shift(waveform, sample_rate, n_steps=4):
        waveform = waveform.numpy()
        waveform_pitch_shifted = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
        waveform_pitch_shifted = torch.from_numpy(waveform_pitch_shifted)
        return waveform_pitch_shifted, sample_rate

    @staticmethod
    def time_stretch(waveform, sample_rate, stretch_factor=1.0):
        waveform = waveform.numpy()
        waveform_stretched = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        waveform_stretched = torch.from_numpy(waveform_stretched)
        return waveform_stretched, sample_rate

    @staticmethod
    def time_shift(waveform, sample_rate, shift_limit):
        _, sig_len = waveform.shape
        shift_amt = int(random.uniform(-shift_limit, shift_limit) * sig_len)
        return waveform.roll(shift_amt), sample_rate

    @staticmethod
    def add_white_noise(waveform, sample_rate, noise_level=0.005):
        noise = torch.randn(waveform.shape) * noise_level
        noisy_signal = waveform + noise
        return noisy_signal.clamp_(-1, 1), sample_rate  # Ensure the signal stays in the -1 to 1 range

    @staticmethod
    def random_gain(waveform, sample_rate, gain_range=(0.5, 1.5)):
        gain = random.uniform(*gain_range)
        return waveform * gain, sample_rate

    @staticmethod
    def apply_augmentation(waveform, sample_rate, augmentation, params):
        if augmentation == 'pitch_shift':
            return AudioAugment.pitch_shift(waveform, sample_rate, **params)
        elif augmentation == 'time_stretch':
            return AudioAugment.time_stretch(waveform, sample_rate, **params)
        # ... other augmentations ...
        else:
            return waveform, sample_rate

    @staticmethod
    def random_augment(waveform, sample_rate, augmentations):
        chosen_augmentation = random.choice(augmentations)
        augmentation_name = chosen_augmentation['name']
        params = chosen_augmentation.get('params', {})
        return AudioAugment.apply_augmentation(waveform, sample_rate, augmentation_name, params)
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
