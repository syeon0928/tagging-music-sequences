import os
import librosa
import random
import torch
import torchaudio
import pandas as pd


class AudioAugmentor:
    def __init__(self, data_path, output_dir, augmentations, n_augmentations=3):
        self.data_path = data_path
        self.output_dir = output_dir
        self.augmentations = augmentations
        self.n_augmentations = n_augmentations
        self.train_original = None
        self.df_aug = None

    def load_data(self):
        self.train_original = pd.read_csv(self.data_path, index_col=0).reset_index(drop=True)

    def perform_augmentations(self, dry_run=False):
        new_rows = []

        for index, row in self.train_original.iterrows():
            filepath = row['filepath']
            full_path = os.path.join('../data', filepath)

            waveform, sample_rate = torchaudio.load(full_path)

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
    def pitch_shift(waveform, sample_rate, n_steps=4):
        waveform_shifted = librosa.effects.pitch_shift(waveform.numpy(), sr=sample_rate, n_steps=n_steps)
        return torch.from_numpy(waveform_shifted), sample_rate

    @staticmethod
    def time_stretch(waveform, sample_rate, stretch_factor=1.0):
        waveform_stretched = librosa.effects.time_stretch(waveform.numpy(), rate=stretch_factor)
        return torch.from_numpy(waveform_stretched), sample_rate

    def apply_augmentation(self, waveform, sample_rate, augmentation, params):
        if augmentation == 'pitch_shift':
            return self.pitch_shift(waveform, sample_rate, **params)
        elif augmentation == 'time_stretch':
            return self.time_stretch(waveform, sample_rate, **params)
        else:
            return waveform, sample_rate

    def random_augment(self, waveform, sample_rate):
        chosen_augmentation = random.choice(self.augmentations)
        return self.apply_augmentation((waveform, sample_rate), chosen_augmentation['name'],
                                       chosen_augmentation.get('params', {}))


def main():
    data_path = '../data/mtat_train_label.csv'
    output_dir = '../data/mtat/augmented'

    # Set a seed for reproducibility
    random.seed(1)

    # Define augmentations
    augmentations = [
        {'name': 'pitch_shift', 'params': {'n_steps': random.randint(-4, 4)}},
        {'name': 'time_stretch', 'params': {'stretch_factor': random.uniform(0.8, 1.25)}}
    ]

    # Number of augmentations per audio file
    n_augmentations = 2

    # Create an instance of the AudioAugmentor
    augmentor = AudioAugmentor(data_path, output_dir, augmentations, n_augmentations=n_augmentations)
    augmentor.load_data()
    augmentor.perform_augmentations(dry_run=False)
    augmented_data_df = augmentor.df_aug

    augmented_data_df.to_csv('../data/mtat_train_label_augmented.csv', index=False)


if __name__ == "__main__":
    main()
