import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np


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

            # Update progress bar after each epoch
            pbar.set_postfix({'epoch': epoch + 1, 'loss': avg_loss / len(self.train_loader)})
            pbar.update(1)

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            self.validate(epoch)

        pbar.close()

    def validate(self, epoch):
        self.model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs)

                all_outputs.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Convert lists to numpy arrays
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        # Threshold for multi-label accuracy
        threshold = 0.5
        predicted = (all_outputs > threshold).astype(int)

        # Multi-label accuracy
        accuracy = np.mean(np.all(predicted == all_labels, axis=1))
        print(f'Epoch [{epoch + 1}], Validation Accuracy: {accuracy:.2f}%')

        # ROC AUC and AUC PR calculations
        # Note: These metrics might not be completely informative for imbalanced multi-label datasets
        roc_auc = roc_auc_score(all_labels, all_outputs, average='macro', multi_class='ovr')
        precision, recall, _ = precision_recall_curve(all_labels.ravel(), all_outputs.ravel())
        auc_pr = auc(recall, precision)

        print(f'ROC AUC: {roc_auc:.2f}, AUC PR: {auc_pr:.2f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
