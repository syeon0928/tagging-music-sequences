import librosa
import numpy as np
import torch


class AudioAugmentation:
    def apply(self, waveform, sample_rate):
        raise NotImplementedError

    def apply_mel(self, mel_spectrogram):
        raise NotImplementedError


class PitchShiftAugmentation(AudioAugmentation):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def apply(self, waveform, sample_rate):
        # librosa expects numpy array, so convert if necessary
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        # Perform pitch shifting
        shifted_waveform = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=self.n_steps)
        return torch.from_numpy(shifted_waveform)


class TimeStretchAugmentation(AudioAugmentation):
    def __init__(self, stretch_factor):
        self.stretch_factor = stretch_factor

    def apply(self, waveform, sample_rate):
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        # Perform time stretching
        stretched_waveform = librosa.effects.time_stretch(waveform_np, rate=self.stretch_factor)
        return torch.from_numpy(stretched_waveform)


class FrequencyMasking(AudioAugmentation):
    def __init__(self, mask_size=10):
        self.mask_size = mask_size

    def apply_mel(self, mel_spectrogram):
        num_mel_channels = mel_spectrogram.size(0)

        f = int(torch.rand(1) * self.mask_size)
        f0 = int(torch.rand(1) * (num_mel_channels - f))

        mask = torch.ones_like(mel_spectrogram)
        mask[f0:f0 + f, :] = 0
        mel_spectrogram = mel_spectrogram * mask

        return mel_spectrogram


class TimeMasking(AudioAugmentation):
    def __init__(self, mask_size=10):
        self.mask_size = mask_size

    def apply_mel(self, mel_spectrogram):
        num_time_steps = mel_spectrogram.size(1)

        t = int(torch.rand(1) * self.mask_size)
        t0 = int(torch.rand(1) * (num_time_steps - t))

        mask = torch.ones_like(mel_spectrogram)
        mask[:, t0:t0 + t] = 0
        mel_spectrogram = mel_spectrogram * mask

        return mel_spectrogram
