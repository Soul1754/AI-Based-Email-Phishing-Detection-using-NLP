import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data_preprocessing import prepare_data
from model import EmailDataset, BertEmailClassifier

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=5):
    """
    Train the BERT model for email classification.
    
    Args:
        model: The BERT model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on (CPU or GPU)
        num_epochs: Number of training epochs
        
    Returns:
        Trained model and training history
    """
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc="Validation")
            
            for batch in val_progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                
                # Update progress bar
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        accuracy = accuracy_score(true_labels, predictions)
        val_accuracies.append(accuracy)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 50)
    
    # Create history dictionary
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }
    
    return model, history

def plot_training_history(history, save_path=None):
    """Plot and optionally save the training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_encodings, test_encodings, train_labels, test_labels, tokenizer = prepare_data(
        args.data_path, max_length=args.max_length
    )
    
    # Create datasets
    train_dataset = EmailDataset(train_encodings, train_labels)
    test_dataset = EmailDataset(test_encodings, test_labels)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size
    )
    
    # Initialize model
    model = BertEmailClassifier(num_classes=2, dropout_rate=args.dropout_rate)
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train model
    trained_model, history = train_model(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        device,
        num_epochs=args.num_epochs
    )
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, 'bert_email_classifier.pt')
    trained_model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save training history
    plot_path = os.path.join(args.model_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model for email classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_dir", type=str, default="/Users/siddhantgaikwad/Developer/College/TY/CS/CP/email_detection/models", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    main(args)