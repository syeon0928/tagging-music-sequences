from torch.utils.data import DataLoader, Dataset, random_split
from src.audio_util import *
import torchaudio

# Custom dataset class (derived from torch)
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.class_columns = df.drop(columns=['filepath', 'data_origin']).columns.to_list()
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

        # Check if specified class columns contain only binary values and are integers
        for column in self.class_columns:
            if not set(self.df[column].unique()).issubset({0, 1}):
                raise ValueError(f"Column {column} contains non-binary values.")
            self.df[column] = self.df[column].astype(int)  # Convert column to int

    # Number of items in dataset
    def __len__(self):
        return len(self.df)

    # Get i'th item in dataset
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with the relative path
        audio_file = self.data_path + '/' + self.df.loc[idx, 'filepath']

        # Retrieve all class label columns for the row
        labels = self.df.loc[idx, self.class_columns].tolist()

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.fix_audio_length(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.mel_spectrogram_with_db(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        # aug_sgram = AudioUtil.spectrogram_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return sgram, labels
