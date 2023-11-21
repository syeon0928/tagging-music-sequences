import librosa
import numpy as np
import torch


class AudioAugmentation:
    def apply(self, waveform, sample_rate):
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
