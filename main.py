import torch
from src.config import Config
from src.audio_dataset import get_dataloader
from src.trainer import Trainer
import src.model_alex as models


def main():
    # Load configuration
    config = Config(
        data_dir="data/",
        train_annotations="mtat_train_label.csv",
        val_annotations="mtat_val_label.csv",
        test_annotations="mtat_test_label.csv",
        sample_rate=16000,
        target_length=29.1,
        batch_size=32,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        apply_transforms=True,
        mel_spec_params={'n_fft': 512, 'hop_length': 256, 'n_mels': 96, 'top_db': 80},
        model_class_name="FullyConvNet4",
        learning_rate=0.001,
        epochs=10,
        model_path="path/to/save/model.pth")

    # Create dataloaders
    train_loader = get_dataloader(
        annotations_file=config.train_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
        transform_params=config.mel_spec_params
    )

    val_loader = get_dataloader(
        annotations_file=config.val_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
        transform_params=config.mel_spec_params
    )

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = getattr(models, config.model_class_name)
    model = model_class().to(device)

    # Initialize the Trainer
    trainer = Trainer(model, train_loader, val_loader, config.learning_rate, device)

    # Run training
    trainer.train(config.epochs)

    # Optionally, save the trained model
    trainer.save_model(config.model_path)

    # Optionally, evaluate the model on the test set
    test_loader = get_dataloader(
        annotations_file=config.test_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_test,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
        transform_params=config.mel_spec_params
    )
    trainer.evaluate(test_loader)


if __name__ == '__main__':
    main()
