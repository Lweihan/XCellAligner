import os
import argparse
import pickle
import numpy as np
import json
from tqdm import tqdm

def compute_global_stats(cache_dir):
    mif_dir = os.path.join(cache_dir, "mif")
    if not os.path.exists(mif_dir):
        print(f"Error: {mif_dir} does not exist.")
        return

    pkl_files = [f for f in os.listdir(mif_dir) if f.endswith('.pkl')]
    if len(pkl_files) == 0:
        print(f"No .pkl files found in {mif_dir}")
        return

    print(f"Found {len(pkl_files)} patches in {mif_dir}. Aggregating cells...")
    
    # Store all values for each channel
    all_channel_values = []
    
    for f in tqdm(pkl_files, desc="Reading mIF features"):
        path = os.path.join(mif_dir, f)
        with open(path, "rb") as file:
            data = pickle.load(file)
            features = data["features"][0].numpy() # shape: [max_cells, channels]
            mask = data["mask"][0].numpy()         # shape: [max_cells]
            
            # Extract only valid cells
            valid_cells = features[mask == 1]
            if len(valid_cells) == 0:
                continue
                
            if len(all_channel_values) == 0:
                num_channels = valid_cells.shape[1]
                all_channel_values = [[] for _ in range(num_channels)]
            
            for c in range(valid_cells.shape[1]):
                all_channel_values[c].append(valid_cells[:, c])
                
    if len(all_channel_values) == 0:
        print("No valid cells found.")
        return
        
    print("Computing 99th percentiles...")
    percentiles = []
    for c in range(len(all_channel_values)):
        if len(all_channel_values[c]) > 0:
            channel_data = np.concatenate(all_channel_values[c])
            p99 = float(np.percentile(channel_data, 99))
            # Avoid divide by zero if max is exactly 0
            if p99 == 0.0:
                p99 = 1e-6
            percentiles.append(p99)
        else:
            percentiles.append(1e-6)
            
    print(f"99th percentiles per channel: {percentiles}")
    
    out_file = os.path.join(cache_dir, "global_norm_stats.json")
    with open(out_file, "w") as f:
        json.dump({"p99_max": percentiles}, f, indent=4)
        
    print(f"Saved stats to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global normalization stats for mIF")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache directory")
    args = parser.parse_args()
    
    compute_global_stats(args.cache_dir)
