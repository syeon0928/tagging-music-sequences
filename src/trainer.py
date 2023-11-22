import time
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score


class Trainer:
    def __init__(self, model, train_loader, valid_loader, learning_rate, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.learning_rate = learning_rate
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.history = {'train_loss': [],
                        'train_roc_auc': [],
                        'train_pr_auc': [],
                        'val_loss': [],
                        'val_roc_auc': [],
                        'val_pr_auc': []
                        }

    def train(self, epochs):
        self.model.train()
        print("Training started")
        start_time = time.time()

        # iterate over epochs
        for epoch in range(epochs):
            total_loss_train = 0
            predicted_labels_train = []
            true_labels_train = []

            # iterate over batches
            for batch, labels in self.train_loader:
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

            # Calculate evaluation metrics from Training phase in current epoch
            avg_loss_train = total_loss_train / len(self.train_loader)
            self.history['train_loss'].append(avg_loss_train)
            train_roc_auc, train_pr_auc = self.performance_metrics(predicted_labels_train, true_labels_train)
            self.history['train_roc_auc'].append(train_roc_auc)
            self.history['train_pr_auc'].append(train_pr_auc)

            # Retrieve evaluation metrics from validation phase in current epoch
            val_loss, val_roc_auc, val_pr_auc = self.evaluate(self.valid_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_roc_auc'].append(val_roc_auc)
            self.history['val_pr_auc'].append(val_pr_auc)

            # Print performance metrics
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} completed in {elapsed_time:.2f} seconds")
            print(f"Training Loss: {avg_loss_train}, Validation Loss: {val_loss}")
            print(f"Training ROC AUC: {train_roc_auc}, Validation ROC AUC: {val_roc_auc}")
            print(f"Training PR AUC: {train_pr_auc}, Validation PR AUC: {val_pr_auc}")

        # Calculate and print total training elapsed time
        total_elapsed_time = time.time() - start_time
        print(f"Total training time: {total_elapsed_time:.2f} seconds")

        return None

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predicted_labels = []
        true_labels = []

        # iterate over batches
        with torch.no_grad():
            for batch, labels in dataloader:
                batch, labels = batch.to(self.device), labels.to(self.device)
                outputs = self.model(batch)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs)

                predicted_labels.append(probabilities.cpu().numpy())
                true_labels.append(labels.cpu().numpy())

        # calculate performance metrics
        avg_loss = total_loss / len(dataloader)
        roc_auc, pr_auc = self.performance_metrics(predicted_labels, true_labels)

        return avg_loss, roc_auc, pr_auc

    # Implement function to retrieve metrics from predicted and true labels
    def performance_metrics(self, predicted_labels, true_labels):
        # Convert lists to numpy arrays
        predicted_labels = np.concatenate(predicted_labels)
        true_labels = np.concatenate(true_labels)

        # Threshold for multi-label accuracy
        threshold = 0.5
        predicted = (predicted_labels > threshold).astype(int)

        # Calculate performance metrics
        # use averaging across labels for ROC AUC, PR AUC
        roc_auc = roc_auc_score(true_labels, predicted_labels, average='macro', multi_class='ovo')
        pr_auc = average_precision_score(true_labels, predicted_labels, average='macro')

        return roc_auc, pr_auc

    def save_model(self, path):
        base_path, filename = os.path.split(path)
        timestamp = time.strftime("%Y%m%d-%H%M")
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{timestamp}{ext}"
        new_path = os.path.join(base_path, new_filename)

        # Create a dictionary to save both the model state dictionary and the class attributes
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, new_path)

    def load_model(self, path):
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
