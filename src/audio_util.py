import math, random
import numpy as np
import torch
import torchaudio
from torchaudio import transforms
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
    def get_audio_channels(aud):
        waveform, sample_rate = aud
        num_channels = waveform.shape[0]  # The shape is (num_channels, num_samples)
        return num_channels

    @staticmethod
    def get_audio_duration(aud):
        waveform, sample_rate = aud

        num_samples = waveform.shape[1]
        duration_seconds = num_samples / sample_rate
        return duration_seconds

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        waveform, sample_rate = aud

        if waveform.shape[0] == new_channel:
            # Nothing to do
            return aud

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
    def resample(aud, new_sample_rate):
        waveform, sample_rate = aud

        if sample_rate == new_sample_rate:
            # Nothing to do
            return aud

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
    def fix_audio_length(aud, max_s):
        waveform, sample_rate = aud
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
    def mel_spectrogram_with_db(aud, n_mels=64, n_fft=1024, hop_len=None):
        waveform, sample_rate = aud
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
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, waveform.T)  # waveform is transposed to shape [time x channels]
        plt.grid(True)
        plt.title('Waveform')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.show()

    @staticmethod
    def plot_spectrogram(spec, sample_rate, title="Spectrogram", ylabel='Frequency (Hz)', xlabel='Time (seconds)', cmap='viridis'):
        # Convert complex tensor to magnitude spectrogram
        spec_mag = torch.abs(spec)

        # Take the magnitude of the complex tensor, and transpose it for display purposes
        spec_mag = spec_mag.squeeze().t()

        plt.figure(figsize=(12, 6))
        plt.imshow(spec_mag.numpy(), origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(label='Magnitude')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mel_spectrogram(mel_spec_db, sample_rate, n_mels=64, n_fft=1024, hop_len=512, title="Mel Spectrogram"):
        # Squeeze the tensor to remove the channel dimension if necessary
        if mel_spec_db.ndim == 3 and mel_spec_db.shape[0] == 1:
            mel_spec_db = mel_spec_db.squeeze(0)

        # Calculate the time axis in seconds
        time_axis = np.linspace(0, mel_spec_db.shape[-1] * hop_len / sample_rate, num=mel_spec_db.shape[-1])

        # Plot the spectrogram
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spec_db, cmap='viridis', aspect='auto', origin='lower', extent=[0, max(time_axis), 0, n_mels])
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Mel Frequency Bins')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()


class AudioAugment:

    @staticmethod
    def time_shift(aud, shift_limit):
        signal, signal_rate = aud
        _, sig_len = signal.shape
        shift_amt = int(random.uniform(-shift_limit, shift_limit) * sig_len)
        return signal.roll(shift_amt), signal_rate

    def pitch_shift(aud, n_steps):
        signal, sample_rate = aud
        signal = signal.numpy()  # Convert to numpy array for librosa compatibility
        shifted_signal = torchaudio.transforms.PitchShift(sample_rate, n_steps)(signal)
        return torch.from_numpy(shifted_signal), sample_rate  # Convert back to torch tensor

    @staticmethod
    def add_white_noise(aud, noise_level=0.005):
        signal, sample_rate = aud
        noise = torch.randn(signal.shape) * noise_level
        noisy_signal = signal + noise
        return noisy_signal.clamp_(-1, 1), sample_rate  # Ensure the signal stays in the -1 to 1 range

    @staticmethod
    def time_stretch(aud, stretch_factor=1.0):
        signal, sample_rate = aud
        # Placeholder: actual time stretching code would go here
        # Since PyTorch doesn't natively support time stretching, we'll return the signal unchanged
        # but with adjusted sample rate to simulate the effect
        new_sample_rate = int(sample_rate / stretch_factor)
        return signal, new_sample_rate  # Return unchanged signal with modified sample rate

    @staticmethod
    def random_gain(aud, gain_range=(0.5, 1.5)):
        signal, sample_rate = aud
        gain = random.uniform(*gain_range)
        return signal * gain, sample_rate