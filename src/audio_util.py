import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        signal, sampling_rate = torchaudio.load(audio_file)
        return (signal, sampling_rate)

    @staticmethod
    def get_audio_channels(audio_file):
        # Load audio file
        signal, sampling_rate = torchaudio.load(audio_file)
        # Get the number of channels
        num_channels = signal.shape[0]  # The shape is (num_channels, num_samples)
        return num_channels

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        signal, sampling_rate = aud

        if (signal.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resampled_signal = signal[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resampled_signal = torch.cat([signal, signal])

        return ((resampled_signal, sampling_rate))

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, new_sampling_rate):
        signal, sampling_rate = aud

        if (sampling_rate == new_sampling_rate):
            # Nothing to do
            return aud

        num_channels = signal.shape[0]
        # Resample first channel
        resampled_signal = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(signal[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            resample_second_channel = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(signal[1:,:])
            resampled_signal = torch.cat([resampled_signal, resample_second_channel])

        return ((resampled_signal, new_sampling_rate))

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        signal, sampling_rate = aud
        num_rows, signal_len = signal.shape
        max_len = sampling_rate//1000 * max_ms

        if (signal_len > max_len):
            # Truncate the signal to the given length
            signal = signal[:,:max_len]

        elif (signal_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            signal = torch.cat((pad_begin, signal, pad_end), 1)

        return (signal, sampling_rate)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        signal, signal_rate = aud
        _, sig_len = signal.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (signal.roll(shift_amt), signal_rate)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def mel_spectrogram_with_db(aud, n_mels=64, n_fft=1024, hop_len=None):
        signal, sampling_rate = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spectrogram = transforms.MelSpectrogram(sampling_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        # Convert to decibels
        spectrogram = transforms.AmplitudeToDB(top_db=top_db)(spectrogram)
        return (spectrogram)

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectrogram_augment(spectrogram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spectrogram.shape
        mask_value = spectrogram.mean()
        aug_spec = spectrogram

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
