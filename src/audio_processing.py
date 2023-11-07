# src/audio_processing.py
import torchaudio
import librosa
import numpy as np

def read_audio(filepath, sample_rate=None):
    # Using torchaudio to load audio
    audio, current_sample_rate = torchaudio.load(filepath)
    if sample_rate is not None and current_sample_rate != sample_rate:
        # Resample if the sample rates do not match
        resample = torchaudio.transforms.Resample(current_sample_rate, sample_rate)
        audio = resample(audio)
    return audio, sample_rate or current_sample_rate

def resample_audio(audio, current_rate, target_rate):
    # Resample audio to a new rate using torchaudio
    if current_rate == target_rate:
        return audio
    resample = torchaudio.transforms.Resample(current_rate, target_rate)
    return resample(audio)

def spectrogram(audio, sample_rate, n_fft=2048, hop_length=512):
    # Generate a spectrogram with torchaudio
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
    return spec(audio)

def mfcc(audio, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    # Compute MFCCs using torchaudio, if torchaudio's MFCC is not satisfactory, you can use librosa
    # Here we use torchaudio's MelSpectrogram and then apply DCT (Discrete Cosine Transform) to get MFCC
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mfcc)
    mel_spectro = mel_spectrogram(audio)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={'n_fft': n_fft, 'n_mels': n_mfcc, 'hop_length': hop_length})
    mfccs = mfcc_transform(mel_spectro)
    return mfccs


# Test for read_audio function
file_path = '../data/raw/gtzan_data/genres_original/classical/classical.00000.wav'
audio, sample_rate = read_audio(file_path, sample_rate=22050)
print(f"Audio Tensor: {audio.shape}, Sample Rate: {sample_rate}")

# Test for resample_audio function
# Assuming original audio is not at 22050 and you want to test resampling
target_sample_rate = 16000  # Target sample rate
resampled_audio = resample_audio(audio, sample_rate, target_sample_rate)
print(f"Resampled Audio Tensor: {resampled_audio.shape}")

# Test for spectrogram function
spec = spectrogram(audio, sample_rate)
print(f"Spectrogram Shape: {spec.size()}")

# Test for mfcc function
mfccs = mfcc(audio, sample_rate)
print(f"MFCCs Shape: {mfccs.size()}")
