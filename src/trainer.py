import time
import datetime
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from src import audio_dataset


class Trainer:
    def __init__(self, 
                 model,
                 annotations_file=None,
                 data_dir=None,
                 batch_size=16,
                 num_workers=0,
                 sample_rate=16000,
                 target_length=29.1,
                 apply_transformations=False,
                 apply_augmentations=False,
                 valid_loader=None,
                 learning_rate=0.001,
                 apply_transfer=False,
                 device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.annotations_file=annotations_file
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.sample_rate=sample_rate
        self.target_length=target_length
        self.apply_transformations=apply_transformations
        self.apply_augmentations=apply_augmentations
        self.valid_loader = valid_loader
        self.learning_rate = learning_rate

        if apply_transfer:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        self.history = {
            "train_loss": [],
            "train_roc_auc": [],
            "train_pr_auc": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
        }

        self.best_val_loss = float('inf')
        self.best_model_state_dict = None
        self.best_history = {}

    def train(self, epochs, save_directory):
        self.model.train()
        print("Training started")
        start_time = time.time()

        # BEGIN TRAINING LOOP
        # iterate over epochs
        for epoch in range(epochs):
            total_loss_train = 0
            predicted_labels_train = []
            true_labels_train = []

            # data loader part
            train_loader = audio_dataset.get_dataloader(
                annotations_file=self.annotations_file,
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                sample_rate=self.sample_rate,
                target_length=self.target_length,
                apply_transformations=self.apply_transformations,
                apply_augmentations=self.apply_augmentations
            )

            # iterate over batches
            for batch, labels, _ in train_loader:
                batch, labels = batch.to(self.device), labels.to(self.device)

                # train
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss_train += loss.item()

                # Apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                predicted_labels_train.append(probabilities)
                true_labels_train.append(labels.cpu().numpy())

            # Retrieve training metrics
            avg_loss_train = total_loss_train / len(train_loader)
            train_roc_auc, train_pr_auc = self.get_auc(predicted_labels_train, true_labels_train)

            # Store train metrics
            self.history["train_loss"].append(avg_loss_train)
            self.history["train_roc_auc"].append(train_roc_auc)
            self.history["train_pr_auc"].append(train_pr_auc)

            # Retrieve validation metrics
            val_loss, val_roc_auc, val_pr_auc = self.evaluate(self.valid_loader, validation=True)

            # Store validation metrics
            self.history["val_loss"].append(val_loss)
            self.history["val_roc_auc"].append(val_roc_auc)
            self.history["val_pr_auc"].append(val_pr_auc)

            # Print performance metrics
            elapsed_time = (datetime.datetime.min + datetime.timedelta(seconds=time.time() - start_time)).strftime(
                "%H:%M:%S")
            print(f"Epoch {epoch + 1}/{epochs} completed in {elapsed_time}")
            print(f"Training Loss: {avg_loss_train}, Validation Loss: {val_loss}")
            print(f"Training ROC AUC: {train_roc_auc}, Validation ROC AUC: {val_roc_auc}")
            print(f"Training PR AUC: {train_pr_auc}, Validation PR AUC: {val_pr_auc}")
            print()

            # Save model state if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state_dict = self.model.state_dict()

                # Update best history
                self.best_history = {
                    "train_loss": self.history["train_loss"].copy(),
                    "train_roc_auc": self.history["train_roc_auc"].copy(),
                    "train_pr_auc": self.history["train_pr_auc"].copy(),
                    "val_loss": self.history["val_loss"].copy(),
                    "val_roc_auc": self.history["val_roc_auc"].copy(),
                    "val_pr_auc": self.history["val_pr_auc"].copy()
                }

        # END OF TRAINING LOOP

        # Save final model
        self.save_model(save_directory, model_type="final")

        # Save the best model
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            self.save_model(save_directory, model_type="best")

        # Calculate and print total training elapsed time
        total_elapsed_time = (datetime.datetime.min + datetime.timedelta(seconds=time.time() - start_time)).strftime(
            "%H:%M:%S")
        print(f"Total training time: {total_elapsed_time:}")

        return None

    def evaluate(self, dataloader, validation=False):
        self.model.eval()
        total_loss = 0
        predicted_labels = []
        true_labels = []
        filepaths = []

        # iterate over batches
        with torch.no_grad():
            for batch, labels, file in dataloader:
                batch, labels = batch.to(self.device), labels.to(self.device)
                outputs = self.model(batch)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs)

                predicted_labels.append(probabilities.cpu().numpy())
                true_labels.append(labels.cpu().numpy())
                filepaths.append(file)

        # calculate performance metrics
        avg_loss = total_loss / len(dataloader)
        roc_auc, pr_auc = self.get_auc(predicted_labels, true_labels)

        if validation:
            return avg_loss, roc_auc, pr_auc
        return avg_loss, roc_auc, pr_auc, predicted_labels, true_labels, filepaths

        # Implement function to retrieve metrics from predicted and true labels

    def get_auc(self, predicted_labels, true_labels):
        # Convert lists to numpy arrays
        predicted_labels = np.concatenate(predicted_labels)
        true_labels = np.concatenate(true_labels)

        # Calculate performance metrics
        # use averaging across labels for ROC AUC, PR AUC
        roc_auc = metrics.roc_auc_score(true_labels, predicted_labels, average="macro", multi_class="ovo")
        pr_auc = metrics.average_precision_score(true_labels, predicted_labels, average="macro")

        return roc_auc, pr_auc

    def save_model(self, directory, model_type="final"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_name = type(self.model).__name__
        timestamp = time.strftime("%Y%m%d-%H%M")

        if model_type == "best":
            filename = f"{model_name}_best_{timestamp}.pth"
            checkpoint = {
                "model_state_dict": self.best_model_state_dict,
                "history": self.best_history,
            }
        elif model_type == "final":
            filename = f"{model_name}_final_{timestamp}.pth"
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "history": self.history
            }
        else:
            filename = f"{model_name}_{model_type}_{timestamp}.pth"
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "history": self.history
            }
        path = os.path.join(directory, filename)
        torch.save(checkpoint, path)

    def load_model(self, path):
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]
