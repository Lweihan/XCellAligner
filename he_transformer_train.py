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

from utils import load_cellpose_model
from dataset import CellDataset
from models import TransformerEncoder
from loss import ContrastiveLoss, InfoNCELoss
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Try to import ctranpath, install if not available
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")

def train_model(data_root, save_path, epochs=100, batch_size=16, learning_rate=1e-5, test_size=0.1, log_path="./log/training_log_11-26.txt"):
    """Train the HE-Transformer model with contrastive learning"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    print("Loading Cellpose model...")
    cellpose_model = load_cellpose_model(model_type='cyto', gpu=torch.cuda.is_available())
    
    print("Loading CTransPath model...")
    ctranspath_model = ctranspath()
    ctranspath_model.to(device)
    
    # Initialize our transformer model
    # CTransPath outputs 1000-dim features
    input_dim = 1000
    hidden_dim = 512
    n_heads = 4
    num_layers = 6
    output_dim = 8  # As defined in models.py
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
    # contrastive_criterion = ContrastiveLoss()
    contrastive_criterion = InfoNCELoss()
    classification_criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = CellDataset(data_root, cellpose_model, ctranspath_model, device, max_cells)
    
    # Check if we have any data
    if len(dataset) == 0:
        print("No cell features extracted. Please check your data.")
        return
    
    # Dynamically fetch number of classes
    num_classes = dataset.get_num_classes()
    print(f"Detected {num_classes} classes.")
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Split dataset into train and test sets
    train_dataset, test_dataset = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)
    
    # Create data loaders for train and test sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_dataset)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_dataset)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Open log file for writing losses
    with open(log_path, "a") as log_file:
        log_file.write("Epoch, Train Contrastive Loss, Train Classification Loss, Test Contrastive Loss, Test Classification Loss\n")  # Header for the log file
    
        # Training loop
        best_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            total_contrastive_loss = 0
            total_classification_loss = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for features_batch, mask_batch, label_batch in progress_bar:
                # Move to device
                features_batch = features_batch.float().to(device)
                mask_batch = mask_batch.float().to(device)
                label_batch = label_batch.to(device)
                
                # Forward pass through transformer for the entire batch
                cls_token, x = model(features_batch, mask_batch)  # x: [batch_size, max_cells, output_dim]

                # Contrastive Learning Loss
                feat1_list, feat2_list, pair_labels = [], [], []
                
                batch_size_actual = x.size(0)
                # for i in range(batch_size_actual):
                #     for j in range(i+1, batch_size_actual):
                #         mask_i = mask_batch[i].bool()
                #         mask_j = mask_batch[j].bool()

                #         valid_features_i = x[i][mask_i]
                #         valid_features_j = x[j][mask_j]

                #         if len(valid_features_i) == 0 or len(valid_features_j) == 0:
                #             continue

                #         valid_labels_i = label_batch[i][mask_i]
                #         valid_labels_j = label_batch[j][mask_j]

                #         for feat_i, feat_j, label_i, label_j in zip(valid_features_i, valid_features_j, valid_labels_i, valid_labels_j):
                #             if label_i == label_j:
                #                 pair_labels.append(1)
                #             else:
                #                 pair_labels.append(0)
                #             feat1_list.append(feat_i)
                #             feat2_list.append(feat_j)
                cell_feats = []
                cell_labels = []
                for i in range(batch_size_actual):
                    mask_i = mask_batch[i].bool()
                    feats_i = x[i][mask_i]               # [num_cells_i, C]
                    labels_i = label_batch[i][mask_i]   # [num_cells_i]

                    cell_feats.append(feats_i)
                    cell_labels.append(labels_i)

                # if len(feat1_list) == 0:
                #     continue

                # feat1_batch = torch.stack(feat1_list)
                # feat2_batch = torch.stack(feat2_list)
                # pair_labels = torch.LongTensor(pair_labels).to(device)
                # Concat all cells across batch
                cell_feats = torch.cat(cell_feats, dim=0)      # [N_cells_total, C]
                cell_labels = torch.cat(cell_labels, dim=0)    # [N_cells_total]

                # contrastive_loss = contrastive_criterion(feat1_batch, feat2_batch, pair_labels)
                contrastive_loss = contrastive_criterion(cell_feats, cell_labels)
                
                # Flatten x and label_batch to compute the classification loss
                x_flat = x.view(-1, x.size(-1))  # Flatten to [batch_size * max_cells, output_dim]
                valid_labels_flat = label_batch.view(-1)  # Flatten the labels

                # Use mask_batch to ignore padding regions (where mask_batch is 0)
                valid_mask_flat = mask_batch.view(-1)  # Flatten mask_batch to get the valid positions

                # Only consider valid cells (where mask_batch is 1)
                x_flat = x_flat[valid_mask_flat == 1]
                valid_labels_flat = valid_labels_flat[valid_mask_flat == 1]

                # Ensure x_flat is float32 (CrossEntropyLoss requires float32)
                x_flat = x_flat.float()
                valid_labels_flat = valid_labels_flat.long()

                # Compute the classification loss only for valid cells
                classification_loss = classification_criterion(x_flat, valid_labels_flat)

                # Total loss (contrastive + classification)
                total_loss = contrastive_loss + classification_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_contrastive_loss += contrastive_loss.item()
                total_classification_loss += classification_loss.item()
                valid_batches += 1
                progress_bar.set_postfix({
                    'contrastive_loss': contrastive_loss.item(),
                    'classification_loss': classification_loss.item()
                })
            
            if valid_batches > 0:
                avg_contrastive_loss = total_contrastive_loss / valid_batches
                avg_classification_loss = total_classification_loss / valid_batches
                print(f'Epoch [{epoch+1}/{epochs}], Average Train Contrastive Loss: {avg_contrastive_loss:.4f}, Average Train Classification Loss: {avg_classification_loss:.4f}')
                
                # Log training loss
                with open(log_path, "a") as log_file:
                    log_file.write(f"{epoch+1}, {avg_contrastive_loss:.4f}, {avg_classification_loss:.4f}, ")

            model.eval()
            total_test_contrastive_loss = 0
            total_test_classification_loss = 0
            valid_test_batches = 0
            with torch.no_grad():
                for features_batch, mask_batch, label_batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} - Evaluating'):
                    features_batch = features_batch.float().to(device)
                    mask_batch = mask_batch.float().to(device)
                    label_batch = label_batch.to(device)

                    cls_token, x = model(features_batch, mask_batch)

                    cell_feats = []
                    cell_labels = []
                    batch_size_actual_test = x.size(0)
                    for i in range(batch_size_actual_test):
                        mask_i = mask_batch[i].bool()
                        feats_i = x[i][mask_i]               # [num_cells_i, C]
                        labels_i = label_batch[i][mask_i]   # [num_cells_i]

                        cell_feats.append(feats_i)
                        cell_labels.append(labels_i)

                    # if len(feat1_list) == 0:
                    #     continue

                    # feat1_batch = torch.stack(feat1_list)
                    # feat2_batch = torch.stack(feat2_list)
                    # pair_labels = torch.LongTensor(pair_labels).to(device)
                    # Concat all cells across batch
                    cell_feats = torch.cat(cell_feats, dim=0)      # [N_cells_total, C]
                    cell_labels = torch.cat(cell_labels, dim=0)    # [N_cells_total]

                    # contrastive_loss = contrastive_criterion(feat1_batch, feat2_batch, pair_labels)
                    contrastive_loss = contrastive_criterion(cell_feats, cell_labels)
                    
                    # Flatten x and label_batch to compute the classification loss
                    x_flat = x.view(-1, x.size(-1))  # Flatten to [batch_size * max_cells, output_dim]
                    valid_labels_flat = label_batch.view(-1)  # Flatten the labels

                    # Use mask_batch to ignore padding regions (where mask_batch is 0)
                    valid_mask_flat = mask_batch.view(-1)  # Flatten mask_batch to get the valid positions

                    # Only consider valid cells (where mask_batch is 1)
                    x_flat = x_flat[valid_mask_flat == 1]
                    valid_labels_flat = valid_labels_flat[valid_mask_flat == 1]

                    # Ensure x_flat is float32 (CrossEntropyLoss requires float32)
                    x_flat = x_flat.float()
                    valid_labels_flat = valid_labels_flat.long()

                    # Compute the classification loss only for valid cells
                    classification_loss = classification_criterion(x_flat, valid_labels_flat)

                    total_test_contrastive_loss += contrastive_loss.item()
                    total_test_classification_loss += classification_loss.item()
                    valid_test_batches += 1

                if valid_test_batches > 0:
                    avg_test_contrastive_loss = total_test_contrastive_loss / valid_test_batches
                    avg_test_classification_loss = total_test_classification_loss / valid_test_batches
                    print(f'Epoch [{epoch+1}/{epochs}], Test Average Contrastive Loss: {avg_test_contrastive_loss:.4f}, Test Average Classification Loss: {avg_test_classification_loss:.4f}')
                    
                    # Log test loss
                    with open(log_path, "a") as log_file:
                        log_file.write(f"{avg_test_contrastive_loss:.4f}, {avg_test_classification_loss:.4f}\n")
                
                # Save the model only if the test loss is improved
                if avg_test_contrastive_loss + avg_test_classification_loss < best_loss:
                    best_loss = avg_test_contrastive_loss + avg_test_classification_loss
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, save_path)
                    print(f"Saved best model with test loss: {best_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HE-Transformer with contrastive learning')
    parser.add_argument('--data_root', type=str, required=True, 
                        help='Path to dataset directory containing images and labels subdirectories')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
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