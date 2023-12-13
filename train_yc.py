import argparse
import random

import torch

import src.models as models
from src.audio_augmentations import PitchShiftAugmentation, TimeStretchAugmentation
from src.audio_dataset import get_dataloader
from src.trainer_yc import Trainer


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

    front_end_model = getattr(models, config.front_end)(
        sample_rate=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels, attention_channels=config.attention_channels)
    back_end_model = getattr(models, config.back_end)(
        config)

    model = model_class(front_end_model, back_end_model).to(device)

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
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=96)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--apply_augmentations", action="store_true")

    parser.add_argument("--model_class_name", type=str, default="FrontEndBackEndModel")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="models")
    #BERTConfig
    parser.add_argument("--attention_channels", default=512)
    parser.add_argument("--attention_layers", default=2)
    parser.add_argument("--attention_heads", default=8)
    parser.add_argument("--attention_length", default=257)
    parser.add_argument("--attention_dropout", default=0.1)

    parser.add_argument("--hidden_size",default= 768)
    parser.add_argument("--num_hidden_layers", default=12)
    parser.add_argument("--num_attention_heads", default=12)
    parser.add_argument("--intermediate_size", default=3072)
    parser.add_argument("--hidden_act", default="gelu")
    parser.add_argument("--max_position_embeddings", default=512)
    parser.add_argument("--attention_probs_dropout_prob",default= 0.1)
    parser.add_argument("--hidden_dropout_prob",default= 0.1)
    parser.add_argument("--type_vocab_size", default=2)

    parser.add_argument('--front_end', type=str, default='FCN5FE', help='Front-end model class name')
    parser.add_argument('--back_end', type=str, default='AttentionModule', help='Back-end model class name')

    config = parser.parse_args()

    print(config)
    main(config)
