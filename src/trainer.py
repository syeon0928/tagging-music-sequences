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
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_roc_auc': [], 'val_pr_auc': []}

        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            history['train_loss'].append(avg_loss)

            # Update progress bar after each epoch
            pbar.set_postfix({'epoch': epoch + 1, 'loss': avg_loss / len(self.train_loader)})
            pbar.update(1)

            # Retrieve values from validation phase in current epoch
            val_loss, val_accuracy, val_roc_auc, val_pr_auc = self.validate(epoch)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_roc_auc'].append(val_roc_auc)
            history['val_pr_auc'].append(val_pr_auc)

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            self.validate(epoch)

        pbar.close()

        return history

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)  # Assuming you have a criterion for validation loss
                total_val_loss += loss.item()

                # Apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs)

                all_outputs.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(self.valid_loader)

        # Convert lists to numpy arrays
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        # Threshold for multi-label accuracy
        threshold = 0.5
        predicted = (all_outputs > threshold).astype(int)

        # Multi-label accuracy
        accuracy = np.mean(np.all(predicted == all_labels, axis=1))

        # Calculate performance metrics
        # use ONE vs ONE approach for ROC AUC as papers suggest to handle class imbalance better
        roc_auc = roc_auc_score(all_labels, all_outputs, average='macro', multi_class='ovo')
        pr_auc = average_precision_score(all_labels, all_outputs, average='macro')
        precision, recall, _ = precision_recall_curve(all_labels.ravel(), all_outputs.ravel())

        print(f'Epoch [{epoch + 1}] Validation Metrics: \n, Validation Accuracy: {accuracy:.2f}%, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}')

        return avg_val_loss, accuracy, roc_auc, pr_auc

    def evaluate(self, test_loader):
        self.model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs)
                all_outputs.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        threshold = 0.5
        predicted = (all_outputs > threshold).astype(int)
        accuracy = np.mean(np.all(predicted == all_labels, axis=1))

        roc_auc = roc_auc_score(all_labels, all_outputs, average='macro', multi_class='ovr')
        precision, recall, _ = precision_recall_curve(all_labels.ravel(), all_outputs.ravel())
        auc_pr = auc(recall, precision)

        # Return a dictionary of metrics
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'auc_pr': auc_pr,
            'all_outputs': all_outputs,
            'all_labels': all_labels
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
