import argparse
import random

import torch

import src.models as models
from src.audio_augmentations import PitchShiftAugmentation, TimeStretchAugmentation
from src.audio_dataset import get_dataloader
from src.trainer import Trainer


def main(config):

    if config.apply_augmentations:
        augmentations = []
        pitch_shift_steps = random.randint(-4, 4)
        time_stretch_factor = random.uniform(0.8, 1.25)
        if config.apply_augmentations:
            augmentations.append(PitchShiftAugmentation(pitch_shift_steps))
            augmentations.append(TimeStretchAugmentation(time_stretch_factor))
    else:
        augmentations = None

    # Create dataloaders
    train_loader = get_dataloader(
        annotations_file=config.train_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
        augmentation=augmentations,
    )

    val_loader = get_dataloader(
        annotations_file=config.val_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
    )

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = getattr(models, config.model_class_name)
    model = model_class().to(device)

    # Initialize the Trainer
    trainer = Trainer(model, train_loader, val_loader, config.learning_rate, device)

    # Run training
    trainer.train(config.epochs, config.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Tagging Training Script")

    # Add arguments for each configuration parameter
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--train_annotations", type=str, default="mtat_train_label.csv")
    parser.add_argument("--val_annotations", type=str, default="mtat_val_label.csv")
    parser.add_argument("--test_annotations", type=str, default="mtat_test_label.csv")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--target_length", type=float, default=29.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--apply_augmentations", action="store_true")

    parser.add_argument("--model_class_name", type=str, default="FCN5")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="models")

    config = parser.parse_args()

    print(config)
    main(config)
