"""
bert_trainer.py

ModernBERT implementation, training a BERT classifier on detecting cluster membership for a given query+document. The implementation has profited from code taken from https://www.kaggle.com/code/dmitrysirenchenko/modernbert-reviews-classification/notebook. Run script with

$python bert_trainer.py --num_epochs <int>

Example usage:

$python bert_trainer.py --num_epochs 3
"""

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import sys
import time
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    """
    Collecting arguments to train a ModernBERT classifier on clusters.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess data."
    )

    parser.add_argument("--num_epochs", type=int, 
    help="Number of training epochs, e.g., 3.")

    return parser.parse_args()


class TextDataset(Dataset):
    """
    Used in torch dataloaders or batches.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BERTTrainer:
    """
    Training class for ModernBERT used as a classifier.
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.n_classes = self.get_classes()
        self.device = self.get_device()
        self.batch_size = 16
        self.model_id = "answerdotai/ModernBERT-base"
        self.model = self.build()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_length = 512
        self.train_loader, self.val_loader = self.load_data()
        self.best_model_path = "model/best_BERT_model.pth" 
    
    def get_classes(self) -> int:
        """ 
        Helper method to get number of classes from data
        """
        return max(self.df["cluster"]) + 1 # Cluster classes range from [0,..,n]

    def get_device(self) -> torch.device:
        """ 
        Helper method to assign self.device according to availability.
        """
        if torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """ 
        Loads data from data/vector_db.jsonl and combines questions and chunks into a single combined text that will serve as the model's training and test data. Labels are the clusters.

        Returns:
            tuple, of DataLoaders
        """
        question = self.df["question"].tolist()
        chunks = self.df["chunk"].tolist()

        # Combining question with answer
        # Chunks from hagrid are lists therefore c[0]
        combined = [" ".join([q, c[0]]) for q, c in zip(question, chunks)] 
        labels = self.df["cluster"].tolist()

        # Train/val split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            combined, labels, test_size=0.1, random_state=42
        )

        # Tokenize with max_length=512 (what is a good parameter here?)
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_length)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.max_length)

        # Create datasets
        train_dataset = TextDataset(train_encodings, train_labels)
        val_dataset = TextDataset(val_encodings, val_labels)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, self.batch_size)

        return train_loader, val_loader

    def build(self) -> nn.Module:
        """
        Instantiates classifier with assigned number of classes.

        Returns:
            classifier based on provided model id
        """
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.n_classes)
        
        return model.to(self.device)
    
    def training_loop(self, num_epochs: int):
        """
        Selecting best model by comparing accuracy scores after each epoch. Best model is saved to self.best_model_path.

        Args:
            num_epochs, to select best model
        """
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = LinearLR(optimizer,
                     start_factor=1.0,
                     end_factor=0.3,
                     total_iters=10)
        best_accuracy = 0.0  # Track the best accuracy
        running_train_loss = 0.0

        for epoch in range(num_epochs):
            start_time = time.time()  # Time at the start of the epoch
            
            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Preparing the data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                # Backward pass
                loss = outputs.loss
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Logging
                if not batch_idx % 10:
                    print(f"Epoch: {epoch+1:04d}/{num_epochs:04d}"
                        f" | Batch "
                        f"{batch_idx:04d}/"
                        f"{len(self.train_loader):04d} | "
                        f"Loss: {loss:.4f}")
            
            train_loss = running_train_loss / len(self.train_loader)
            print(f"Epoch: {epoch+1}/{num_epochs} | "
            f"Training Loss: {train_loss:.4f} | ")

            # Evaluation after epoch
            self.model.eval()

            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch in self.val_loader:
                    # Preparing data
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(labels.numpy())

            # Time at the end of the epoch
            end_time = time.time()
            epoch_duration = end_time - start_time
                    
            accuracy = accuracy_score(val_true, val_preds)
            print(f"Epoch {epoch + 1}/{num_epochs}, {epoch_duration:.2f} sec. Test Accuracy: {accuracy:.4f}")

            os.makedirs("model", exist_ok=True)  # Ensuring /model path exists

            # Save the model if the accuracy improves
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved with accuracy: {best_accuracy:.4f}")
    
    def evaluation(self):
        """
        Evaluation on the validation dataset: precision, recall, F1.
        """
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))  # Load the best model
        self.model.eval()

        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in self.val_loader:
                #Preparing data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                #Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy())

        print(classification_report(val_true, val_preds, zero_division=0))


if __name__ == "__main__":
    """ 
    Example usage of training a ModernBERT model for classifying input queries.

    Setup: ModernBERT trained with AdamW, lr=5e-5, and a linear scheduler for 4 epochs
    """
    args = parse_args()
    num_epochs = args.num_epochs
    # Model should be only trained with the train split
    df = pd.read_json("data/vector_db.jsonl", lines=True)
    df_train = df[df["split"]=="train"]

    # Instantiate model and train and evaluate it
    model = BERTTrainer(df_train)
    if num_epochs is None:
        num_epochs = 1
    model.training_loop(num_epochs)

    # Evaluation is on question chunk concatenations, not on questions alone
    model.evaluation()
