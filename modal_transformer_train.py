import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from collections import defaultdict
import argparse
from tqdm import tqdm
import random

from dataset import ModalDataset
from models import TransformerEncoder
from loss import InfoNCELoss
from sklearn.model_selection import train_test_split

def train_model(data_root, save_path, epochs=100, batch_size=16, learning_rate=1e-4, test_size=0.1, log_path="./log/modal_training_log.txt"):
    """Train the Modal-Transformer model with contrastive learning only"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset first
    dataset = ModalDataset(data_root, max_cells=255)
    
    # Check if we have any data
    if len(dataset) == 0:
        print("No data found. Please check your data directory.")
        return
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Initialize our transformer model
    # Number of input channels depends on the number of images per sample
    # We need to determine this from the first sample
    temp_loader = DataLoader(dataset, batch_size=1)
    sample_batch = next(iter(temp_loader))
    input_dim = sample_batch[0].size(-1)  # Number of channels
    print(f"Detected {input_dim} input channels")
    
    hidden_dim = 512
    n_heads = 4
    num_layers = 6
    output_dim = 8  # Output dimension for contrastive learning
    max_cells = 255
    
    model = TransformerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        max_cells=max_cells
    ).to(device)
    
    # Loss and optimizer
    contrastive_criterion = InfoNCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Split dataset into train and test sets
    train_dataset, test_dataset = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)
    
    # Create data loaders for train and test sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_dataset)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_dataset)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Open log file for writing losses
    with open(log_path, "a") as log_file:
        log_file.write("Epoch, Train Contrastive Loss, Test Contrastive Loss\n")  # Header for the log file
    
        # Training loop
        best_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            total_contrastive_loss = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for features_batch, mask_batch, label_batch in progress_bar:
                # Move to device
                features_batch = features_batch.float().to(device)
                mask_batch = mask_batch.float().to(device)
                
                # Forward pass through transformer for the entire batch
                cls_token, x = model(features_batch, mask_batch)  # x: [batch_size, max_cells, output_dim]

                # Prepare cell features and labels for contrastive loss
                cell_feats = []
                cell_labels = []
                batch_size_actual = x.size(0)
                for i in range(batch_size_actual):
                    mask_i = mask_batch[i].bool()
                    feats_i = x[i][mask_i]               # [num_cells_i, C]
                    # Create pseudo labels based on batch index for contrastive learning
                    labels_i = torch.full((feats_i.size(0),), i, device=feats_i.device)  # All cells in same batch get same label

                    cell_feats.append(feats_i)
                    cell_labels.append(labels_i)

                # Skip if no cells detected
                if len(cell_feats) == 0 or all(f.size(0) == 0 for f in cell_feats):
                    continue

                # Concat all cells across batch
                cell_feats = torch.cat(cell_feats, dim=0)      # [N_cells_total, C]
                cell_labels = torch.cat(cell_labels, dim=0)    # [N_cells_total]

                # Compute contrastive loss
                contrastive_loss = contrastive_criterion(cell_feats, cell_labels)
                
                optimizer.zero_grad()
                contrastive_loss.backward()
                optimizer.step()

                total_contrastive_loss += contrastive_loss.item()
                valid_batches += 1
                progress_bar.set_postfix({
                    'contrastive_loss': contrastive_loss.item()
                })
            
            if valid_batches > 0:
                avg_contrastive_loss = total_contrastive_loss / valid_batches
                print(f'Epoch [{epoch+1}/{epochs}], Average Train Contrastive Loss: {avg_contrastive_loss:.4f}')
                
                # Log training and test loss for this epoch
                with open(log_path, "a") as log_file:
                    log_file.write(f"{epoch+1}, {avg_contrastive_loss:.4f}, ")
            
            # Evaluation
            model.eval()
            total_test_contrastive_loss = 0
            valid_test_batches = 0
            with torch.no_grad():
                for features_batch, mask_batch, label_batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} - Evaluating'):
                    features_batch = features_batch.float().to(device)
                    mask_batch = mask_batch.float().to(device)

                    cls_token, x = model(features_batch, mask_batch)

                    # Prepare cell features and labels for contrastive loss
                    cell_feats = []
                    cell_labels = []
                    batch_size_actual_test = x.size(0)
                    for i in range(batch_size_actual_test):
                        mask_i = mask_batch[i].bool()
                        feats_i = x[i][mask_i]               # [num_cells_i, C]
                        # Create pseudo labels based on batch index for contrastive learning
                        labels_i = torch.full((feats_i.size(0),), i, device=feats_i.device)  # All cells in same batch get same label

                        cell_feats.append(feats_i)
                        cell_labels.append(labels_i)

                    # Skip if no cells detected
                    if len(cell_feats) == 0 or all(f.size(0) == 0 for f in cell_feats):
                        continue

                    # Concat all cells across batch
                    cell_feats = torch.cat(cell_feats, dim=0)      # [N_cells_total, C]
                    cell_labels = torch.cat(cell_labels, dim=0)    # [N_cells_total]

                    # Compute contrastive loss
                    contrastive_loss = contrastive_criterion(cell_feats, cell_labels)

                    total_test_contrastive_loss += contrastive_loss.item()
                    valid_test_batches += 1

                if valid_test_batches > 0:
                    avg_test_contrastive_loss = total_test_contrastive_loss / valid_test_batches
                    print(f'Epoch [{epoch+1}/{epochs}], Test Average Contrastive Loss: {avg_test_contrastive_loss:.4f}')
                    
                    # Log test loss and complete the line
                    with open(log_path, "a") as log_file:
                        log_file.write(f"{avg_test_contrastive_loss:.4f}\n")
                
                    # Save the model only if the test loss is improved
                    if avg_test_contrastive_loss < best_loss:
                        best_loss = avg_test_contrastive_loss
                        best_model_state = model.state_dict()
                        torch.save(best_model_state, save_path)
                        print(f"Saved best model with test loss: {best_loss:.4f}")
                else:
                    # If no valid test batches, complete the line with a placeholder
                    with open(log_path, "a") as log_file:
                        log_file.write("N/A\n")
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Modal-Transformer with contrastive learning')
    parser.add_argument('--data_root', type=str, required=True, 
                        help='Path to dataset directory containing subdirectories for each image')
    parser.add_argument('--save_path', type=str, default='modal_best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_root=args.data_root,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )