import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score


class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        self.model.train()
        pbar = tqdm(total=epochs, desc='Training', leave=True)
        history = {'train_loss': [], 'train_accuracy': [], 'train_roc_auc': [], 'train_pr_auc': [], 'val_loss': [],
                   'val_accuracy': [], 'val_roc_auc': [], 'val_pr_auc': []}

        # iterate over epochs
        for epoch in range(epochs):
            total_loss_train = 0
            predicted_labels_train = []
            true_labels_train = []

            # iterate over batches
            for batch, labels in self.train_loader:
                batch, labels = batch.to(self.device), labels.to(self.device)

                # traom
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss_train += loss.item()

            # Calculate evaluation metrics from Training phase in current epoch
            avg_loss_train = total_loss_train / len(self.train_loader)
            history['train_loss'].append(avg_loss_train)
            train_accuracy, train_roc_auc, train_pr_auc = self.performance_metrics(predicted_labels_train,
                                                                                   true_labels_train)
            history['train_accuracy'].append(train_accuracy)
            history['train_roc_auc'].append(train_roc_auc)
            history['train_pr_auc'].append(train_pr_auc)

            # Retrieve evaluation metrics from validation phase in current epoch
            val_loss, val_accuracy, val_roc_auc, val_pr_auc = self.evaluate(self.valid_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_roc_auc'].append(val_roc_auc)
            history['val_pr_auc'].append(val_pr_auc)

            # Update progress bar after each epoch
            pbar.set_postfix({'epoch': epoch + 1, 'training loss': avg_loss_train, 'validation loss': val_loss})
            pbar.update(1)

        pbar.close()

        return history

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
        accuracy, roc_auc, pr_auc = self.performance_metrics(predicted_labels, true_labels)

        return avg_loss, accuracy, roc_auc, pr_auc

    # Implement function to retrieve metrics from predicted and true labels
    def performance_metrics(self, predicted_labels, true_labels):
        # Convert lists to numpy arrays
        predicted_labels = np.concatenate(predicted_labels)
        true_labels = np.concatenate(true_labels)

        # Threshold for multi-label accuracy
        threshold = 0.5
        predicted = (predicted_labels > threshold).astype(int)

        # Multi-label accuracy
        accuracy = np.mean(np.all(predicted == true_labels, axis=1))

        # Calculate performance metrics
        # use averaging across labels for ROC AUC, PR AUC
        roc_auc = roc_auc_score(true_labels, predicted_labels, average='macro', multi_class='ovo')
        pr_auc = average_precision_score(true_labels, predicted_labels, average='macro')

        return accuracy, roc_auc, pr_auc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))