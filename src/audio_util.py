import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


class AudioUtil:
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
