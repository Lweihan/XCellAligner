import numpy as np
import torch
from cellpose import models
from PIL import Image


class CellDensityExtractor:
    """
    Component for extracting cell density features from image pairs.
    
    This component takes n pairs of images, each with k images and a list of length k.
    For the first image in each pair, it uses Cellpose to extract nuclei centers.
    For subsequent images, it extracts the proportion of pixels with grayscale values > 0.5 
    within square regions centered at the nuclei centers, using radii defined by nuclei_diam 
    and cell_diam based on whether the corresponding list[idx] is 0 or 1.
    Finally, it organizes the proportions into an m*k tensor where m is the number of cells.
    """
    
    def __init__(self, model_type='nuclei', gpu=True, nuclei_diam=10, cell_diam=20):
        """
        Initialize the CellDensityExtractor.
        
        Args:
            model_type (str): Type of Cellpose model ('nuclei' or 'cyto')
            gpu (bool): Whether to use GPU for Cellpose
            nuclei_diam (int): Radius for nuclei region extraction
            cell_diam (int): Radius for cell region extraction
        """
        self.cellpose_model = models.CellposeModel(model_type=model_type, gpu=gpu)
        self.nuclei_diam = nuclei_diam
        self.cell_diam = cell_diam
    
    def extract_nuclei_centers(self, image, cell_masks):
        """
        Extract nuclei centers using Cellpose.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: List of (y, x) coordinates for nuclei centers
        """
        # Run Cellpose segmentation
        if cell_masks is None:
            cp_results = self.cellpose_model.eval(image, diameter=None, channels=[0, 0])
            masks = cp_results[0]
        else:
            masks = cell_masks
        
        # Find unique cell IDs (excluding background)
        unique_masks = np.unique(masks)
        cell_ids = unique_masks[unique_masks != 0]
        
        # Calculate center of mass for each cell
        centers = []
        for cell_id in cell_ids:
            cell_mask = masks == cell_id
            coords = np.where(cell_mask)
            if len(coords[0]) > 0:
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                centers.append((center_y, center_x))
                
        return centers, masks
    
    def calculate_density_in_region(self, image, center_y, center_x, radius):
        """
        Calculate the proportion of pixels with grayscale > 0.5 in a square region.
        
        Args:
            image (np.ndarray): Input image
            center_y (int): Y coordinate of center
            center_x (int): X coordinate of center
            radius (int): Radius of square region
            
        Returns:
            float: Proportion of pixels with grayscale > 0.5
        """
        # Define square region boundaries
        min_y = max(0, center_y - radius)
        max_y = min(image.shape[0], center_y + radius + 1)
        min_x = max(0, center_x - radius)
        max_x = min(image.shape[1], center_x + radius + 1)
        
        # Extract region
        region = image[min_y:max_y, min_x:max_x]
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            # Simple grayscale conversion (mean of RGB channels)
            region_gray = np.mean(region, axis=2)
        else:
            region_gray = region
        
        threshold = 0.5
        above_threshold_mask = region_gray > threshold
        weighted_sum = np.sum(region_gray[above_threshold_mask])
        total_pixels = region_gray.size
        total_sum = total_pixels * 255
        
        if total_sum == 0:
            return 0.0
        result = weighted_sum / total_sum
        
        return result
        
    def process_image_pair(self, images, flags, cell_masks=None):
        """
        Process a pair of images to extract cell density features.
        
        Args:
            images (list): List of k images (each as np.ndarray)
            flags (list): List of k flags (0 or 1) indicating region type
            
        Returns:
            np.ndarray: Array of shape (m, k) where m is number of cells
        """
        if len(images) != len(flags):
            raise ValueError("Number of images must match number of flags")
            
        # Extract nuclei centers from the first image
        centers, masks = self.extract_nuclei_centers(images[0], cell_masks)
        m = len(centers)  # Number of cells
        
        if m == 0:
            # Return empty array if no cells detected
            return np.empty((0, len(images)))
        
        k = len(images)  # Number of images per pair
        density_matrix = np.zeros((m, k))
        
        # Process each image
        for img_idx, (image, flag) in enumerate(zip(images, flags)):
            # Determine radius based on flag
            radius = self.nuclei_diam if flag == 0 else self.cell_diam
            
            # Calculate density for each cell center
            for cell_idx, (center_y, center_x) in enumerate(centers):
                density = self.calculate_density_in_region(image, center_y, center_x, radius)
                density_matrix[cell_idx, img_idx] = density
                
        return density_matrix
    
    def process_multiple_pairs(self, image_pairs, flags_list):
        """
        Process multiple image pairs.
        
        Args:
            image_pairs (list): List of n image pairs, each containing k images
            flags_list (list): List of n flag lists, each of length k
            
        Returns:
            list: List of np.ndarrays, each of shape (m, k)
        """
        results = []
        for images, flags in zip(image_pairs, flags_list):
            result = self.process_image_pair(images, flags)
            results.append(result)
            
        return results


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


# Example usage:
# extractor = CellDensityExtractor(nuclei_diam=10, cell_diam=20)
# 
# # Load images
# image1 = load_image_as_array('path/to/image1.png')  # First image for nuclei detection
# image2 = load_image_as_array('path/to/image2.png')  # Second image
# image3 = load_image_as_array('path/to/image3.png')  # Third image
# 
# images = [image1, image2, image3]
# flags = [0, 1, 1]  # 0 for nuclei radius, 1 for cell radius
# 
# # Process the image set
# density_tensor = extractor.process_image_pair(images, flags)
# print(f"Density tensor shape: {density_tensor.shape}")  # (m, 3) where m is number of cells