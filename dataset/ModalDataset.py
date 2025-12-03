import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from module.ModalEncoder.cell_density_extractor import CellDensityExtractor

def load_image_as_array(image_path):
    """
    Load an image from path as numpy array.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        np.ndarray: Image as numpy array
    """
    img = Image.open(image_path)
    return np.array(img)

class ModalDataset(Dataset):
    """
    Dataset for multi-modal cell data.
    Each sample contains multiple images of the same tissue from different channels/modalities.
    """
    
    def __init__(self, data_root, max_cells=255):
        """
        Initialize the dataset.
        
        Args:
            data_root (str): Path to the dataset directory containing subdirectories for each image
            max_cells (int): Maximum number of cells per image
        """
        self.data_root = data_root
        self.max_cells = max_cells
        self.image_dirs = [os.path.join(data_root, d) for d in sorted(os.listdir(data_root)) 
                          if os.path.isdir(os.path.join(data_root, d))]
        self.cell_extractor = CellDensityExtractor()
        
        # Collect all samples
        self.samples = []
        self._prepare_samples()
        
        # Pre-extract all features
        self.all_features = []
        self.all_masks = []
        self.all_labels = []
        self._extract_all_features()
        
    def _prepare_samples(self):
        """
        Prepare samples by scanning the data directory.
        """
        # 获取所有目录中的图像文件
        all_image_files_per_dir = []
        for img_dir in self.image_dirs:
            # 获取目录中所有图像文件
            image_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            all_image_files_per_dir.append((img_dir, image_files))
        
        # 确定最小的图像数量（避免索引越界）
        min_images = min(len(files) for _, files in all_image_files_per_dir) if all_image_files_per_dir else 0
        
        # 按索引位置分组
        for i in range(min_images):
            image_paths = []
            dir_names = []
            
            # 收集每个目录中第i个图像
            for img_dir, image_files in all_image_files_per_dir:
                if i < len(image_files):
                    image_paths.append(os.path.join(img_dir, image_files[i]))
                    dir_names.append(img_dir)
            
            if len(image_paths) < 2:
                print(f"Skipping group {i} - needs at least 2 images, got {len(image_paths)}")
                continue
            
            # For flags, we'll use 0 for the first image (nuclei detection) and 1 for others (cell density)
            flags = [0] + [1] * (len(image_paths) - 1)
            
            self.samples.append({
                'image_paths': image_paths,
                'flags': flags,
                'group_index': i,
                'dir_names': dir_names
            })
    
    def _extract_all_features(self):
        """
        Pre-extract features for all samples and store them
        """
        print("Extracting cell features...")
        
        # 进度条应以图像组为单位进行更新
        total_samples = len(self.samples)
        processed_samples = 0
        
        # 创建一个全局的进度条
        pbar = tqdm(total=total_samples, desc="Processing samples")
        
        for sample in self.samples:
            try:
                image_paths = sample['image_paths']
                flags = sample['flags']
                group_index = sample.get('group_index', 0)
                dir_names = sample.get('dir_names', [])
                
                # Load images
                images = [load_image_as_array(path) for path in image_paths]
                
                # Extract cell density features
                density_matrix = self.cell_extractor.process_image_pair(images, flags)
                
                # Convert to tensor
                num_cells = density_matrix.shape[0]
                
                # Handle case with no cells detected
                if num_cells == 0:
                    # Create empty tensors
                    padded_features = torch.zeros((self.max_cells, len(images)))
                    mask = torch.zeros(self.max_cells)
                    labels = torch.zeros(self.max_cells, dtype=torch.long)
                    
                    self.all_features.append(padded_features)
                    self.all_masks.append(mask)
                    self.all_labels.append(labels)
                    # 更新进度条
                    processed_samples += 1
                    pbar.update(1)
                    continue
                
                # Pad or truncate to max_cells
                if num_cells > self.max_cells:
                    # Truncate
                    density_tensor = torch.FloatTensor(density_matrix[:self.max_cells, :])
                    mask = torch.ones(self.max_cells)
                else:
                    # Pad with zeros
                    padding_size = self.max_cells - num_cells
                    density_tensor = torch.FloatTensor(np.pad(density_matrix, ((0, padding_size), (0, 0)), mode='constant'))
                    mask = torch.cat([torch.ones(num_cells), torch.zeros(padding_size)])
                
                # Create labels based on the channel with highest density (excluding the first channel)
                if num_cells > 0:
                    # For each cell, find the channel (excluding the first one) with the highest density value
                    if density_matrix.shape[1] > 1:  # If we have more than one channel
                        # Consider only channels 1 and above (exclude channel 0)
                        non_first_channels = density_matrix[:, 1:]
                        # Find the index of the channel with maximum density for each cell
                        max_density_indices = np.argmax(non_first_channels, axis=1)
                        # Add 1 to the indices because we excluded the first channel
                        labels_raw = max_density_indices + 1
                    else:
                        # If there's only one channel, all labels are 0
                        labels_raw = np.zeros(num_cells, dtype=np.int64)
                        
                    # Convert to tensor and pad/truncate as needed
                    if num_cells > self.max_cells:
                        # Truncate
                        labels = torch.LongTensor(labels_raw[:self.max_cells])
                    else:
                        # Pad with zeros
                        labels = torch.LongTensor(np.pad(labels_raw, (0, padding_size), mode='constant'))
                else:
                    labels = torch.zeros(self.max_cells, dtype=torch.long)
                
                self.all_features.append(density_tensor)
                self.all_masks.append(mask)
                self.all_labels.append(labels)
                
                # 更新进度条
                processed_samples += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing group {group_index}: {e}")
                # Create empty tensors for failed samples
                num_channels = len(sample['image_paths']) if 'image_paths' in sample else 1
                padded_features = torch.zeros((self.max_cells, num_channels))
                mask = torch.zeros(self.max_cells)
                labels = torch.zeros(self.max_cells, dtype=torch.long)
                
                self.all_features.append(padded_features)
                self.all_masks.append(mask)
                self.all_labels.append(labels)
                
                # 更新进度条
                processed_samples += 1
                pbar.update(1)
                continue
        
        # 关闭进度条
        pbar.close()
    
    def __len__(self):
        return len(self.all_features)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the pre-extracted features.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (features, mask, label) where:
                - features: tensor of shape (max_cells, num_channels)
                - mask: tensor of shape (max_cells,) indicating valid cells
                - label: tensor of shape (max_cells,) with class labels based on channel with highest density
        """
        return self.all_features[idx], self.all_masks[idx], self.all_labels[idx]