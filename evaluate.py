import os
import argparse
import torch
import pickle
import src.models as models
from src.audio_dataset import get_dataloader
from src.trainer import Trainer

def evaluate_model(model, model_path, test_loader, device):
    trainer = Trainer(model, train_loader=None, valid_loader=None, learning_rate=0.001, device=device)
    trainer.load_model(model_path)
    return trainer.evaluate(test_loader)

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model_class = getattr(models, config.model_class_name)
    model = model_class().to(device)

    # Initialize the test data loader
    test_loader = get_dataloader(
        annotations_file=config.test_annotations,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        sample_rate=config.sample_rate,
        target_length=config.target_length,
    )

    # Evaluate the model
    evaluation_results = evaluate_model(model, config.model_path, test_loader, device)

    # Print results
    avg_loss, roc_auc, pr_auc, predicted_labels, true_labels, filepaths = evaluation_results

    print(f"Average Loss: {avg_loss}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"PR AUC Score: {pr_auc}")

    # Save results to a file
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    model_name = os.path.basename(config.model_path).split('.')[0]
    results_filename = os.path.join(config.results_dir, f"results-{model_name}.pkl")

    # Save results to a file
    with open(results_filename, 'wb') as file:
        pickle.dump(evaluation_results, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Tagging Model Evaluation Script")

    # Define arguments
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--test_annotations", type=str, default="mtat_test_label.csv")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--results_dir", type=str, default="evaluation_results/", help="Path to store evaluation results")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--target_length", type=float, default=29.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--model_class_name", type=str, help="Model class to evaluate")

    config = parser.parse_args()
    print(config)
    main(config)

