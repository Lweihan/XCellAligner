import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import extract_cell_features


class CellDataset(Dataset):
    def __init__(self, data_root, cellpose_model, ctranspath_model, device, max_cells=255):
        self.data_root = data_root
        self.cellpose_model = cellpose_model
        self.ctranspath_model = ctranspath_model
        self.device = device
        self.max_cells = max_cells

        self.image_dir = os.path.join(data_root, 'images')
        self.label_dir = os.path.join(data_root, 'labels')

        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Flatten the features and masks across all images
        self.all_features = []
        self.all_masks = []
        self.all_labels = []
        self._extract_all_features()

    def _extract_all_features(self):
        """Extract features for all cells in all images and flatten the lists"""
        print("Extracting cell features...")
        for image_file in tqdm(self.image_files):
            image_path = os.path.join(self.image_dir, image_file)
            label_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            label_path = os.path.join(self.label_dir, label_file)

            if not os.path.exists(label_path):
                continue

            # Extract cell features with labels
            try:
                cell_features, cell_labels, masks = extract_cell_features(
                    image_path, label_path, self.cellpose_model, self.ctranspath_model, self.device)

                # Skip if no cells detected
                if len(cell_features) == 0 or len(cell_labels) == 0:
                    print(f"No cells detected in {image_file}, skipping...")
                    continue

                # Limit to max_cells
                if len(cell_features) > self.max_cells:
                    cell_features = cell_features[:self.max_cells]
                    cell_labels = cell_labels[:self.max_cells]

                # Pad or truncate to max_cells and store as single sample
                padded_features = np.zeros((self.max_cells, 1000))  # Shape: (max_cells, feature_dim)
                mask = np.zeros(self.max_cells)  # Shape: (max_cells, )

                num_cells = min(len(cell_features), self.max_cells)
                for i in range(num_cells):
                    padded_features[i] = cell_features[i]
                    mask[i] = 1  # Mark as valid

                # Pad the labels to max_cells
                padded_labels = np.zeros(self.max_cells)
                padded_labels[:num_cells] = cell_labels[:num_cells]

                # Store the features, masks, and labels for all images
                self.all_features.append(padded_features)
                self.all_masks.append(mask)
                self.all_labels.append(padded_labels)

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue

    def get_num_classes(self):
        """Dynamically calculate the number of classes from all labels"""
        all_labels = np.concatenate(self.all_labels)
        num_classes = int(np.max(all_labels)) + 1  # Assuming classes are 0-indexed
        return num_classes

    def __getitem__(self, idx):
        # Return a sample for training
        return self.all_features[idx], self.all_masks[idx], self.all_labels[idx]

    def __len__(self):
        return len(self.all_features)