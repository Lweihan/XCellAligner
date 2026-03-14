import os
import re
import argparse
import pickle
import random
import logging
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from XCellFormer import XCellFormer
#from updated_models import TransformerEncoder


# =========================
# 日志系统
# =========================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TRAIN")


# =========================
# 工具函数
# =========================
def parse_xy(filename):
    m = re.search(r"x(\d+)_y(\d+)", filename)
    if m is None:
        raise ValueError(f"非法文件名: {filename}")
    return int(m.group(1)), int(m.group(2))


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_mif_feature(feat, target_dim):
    if len(feat.shape) == 3:
        if feat.shape[-1] >= target_dim:
            return feat
        pad_size = target_dim - feat.shape[-1]
        pad = torch.zeros(feat.size(0), feat.size(1), pad_size, device=feat.device)
        return torch.cat([feat, pad], dim=-1)
    else:
        raise ValueError(f"Unexpected feature shape: {feat.shape}")


def hungary_mse_loss(he_feat, mif_feat, start_index, mif_channel):
    he_slice = he_feat[:, start_index:start_index + mif_channel]  # Slice HE features
    mif_slice = mif_feat[:, :mif_channel]  # Slice MIF features
    return F.mse_loss(he_slice, mif_slice)


# =========================
# Dataset
# =========================
class HeMifDataset(Dataset):
    def __init__(self, he_dir, mif_dir, num_neg_samples=3):
        self.he_dir = he_dir
        self.mif_dir = mif_dir
        self.num_neg_samples = num_neg_samples

        self.he_files = [f for f in os.listdir(he_dir) if f.endswith(".pkl")]

        self.mif_map = {}
        for f in os.listdir(mif_dir):
            if f.endswith(".pkl"):
                x, y = parse_xy(f)
                self.mif_map[(x, y)] = os.path.join(mif_dir, f)

        self.valid_pairs = []
        for f in self.he_files:
            x, y = parse_xy(f)
            if (x, y) in self.mif_map:
                self.valid_pairs.append((f, (x, y)))

        self.all_coords = list(self.mif_map.keys())
        
        assert len(self.valid_pairs) > 0
        assert len(self.all_coords) > 1

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        he_file, (x, y) = self.valid_pairs[idx]

        he = load_pkl(os.path.join(self.he_dir, he_file))
        mif_pos_raw = load_pkl(self.mif_map[(x, y)])

        neg_xy = self._get_farthest_negative_samples(x, y, self.num_neg_samples)
        mif_neg_raw = [load_pkl(self.mif_map[neg]) for neg in neg_xy]

        # Return the positive and multiple negative samples
        return {
            "he_features": he["features"][0],
            "he_mask": he["mask"][0],
            "mif_pos_features": mif_pos_raw["features"][0],
            "mif_pos_mask": mif_pos_raw["mask"][0],
            "mif_neg_features": [neg["features"][0] for neg in mif_neg_raw],
            "mif_neg_mask": [neg["mask"][0] for neg in mif_neg_raw],
        }

    def _get_farthest_negative_samples(self, curr_x, curr_y, num_neg_samples=3, min_distance_ratio=0.3):
        distances = []
        for x, y in self.all_coords:
            if (x, y) != (curr_x, curr_y):
                dist = ((x - curr_x) ** 2 + (y - curr_y) ** 2) ** 0.5
                distances.append(((x, y), dist))

        if not distances:
            other_coords = [coord for coord in self.all_coords if coord != (curr_x, curr_y)]
            return random.sample(other_coords, num_neg_samples) if other_coords else random.sample(self.all_coords, num_neg_samples)

        distances.sort(key=lambda item: item[1], reverse=True)
        max_dist = distances[0][1]
        min_required_dist = max_dist * min_distance_ratio
        far_coords = [coord for coord, dist in distances if dist >= min_required_dist]

        if len(far_coords) >= num_neg_samples:
            return random.sample(far_coords, num_neg_samples)
        else:
            other_coords = [coord for coord in self.all_coords if coord != (curr_x, curr_y)]
            return random.sample(other_coords, num_neg_samples)


# =========================
# 对比损失
# =========================
def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarity
    pos = (anchor * positive).sum(dim=-1, keepdim=True)

    # Negative similarity
    neg = (anchor.unsqueeze(1) * negatives).sum(dim=-1)

    # Combine all logits
    logits = torch.cat([pos, neg], dim=1) / temperature

    # Labels: first column is positive, rest are negative
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits, labels)


# =========================
# 主训练函数
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(
        args.output_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(os.path.join(exp_dir, "logs"))
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))

    dataset = HeMifDataset(
        os.path.join(args.cache_dir, "he"),
        os.path.join(args.cache_dir, "mif"),
        num_neg_samples=args.num_neg_samples
    )

    # Split dataset into training and testing (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    he_model = XCellFormer(
        input_dim=768,
        hidden_dim=512,
        n_heads=8,
        num_layers=4,
        output_dim=20,
        max_cells=255,
        use_large_vit=False
    ).to(device)

    if args.he_model_path != None and os.path.exists(args.he_model_path):
        he_model.load_state_dict(torch.load(args.he_model_path, map_location=device))

    optimizer = AdamW(
        he_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    global_step = 0
    best_loss = float('inf')
    best_model_path = None

    for epoch in range(args.epochs):
        he_model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_contrast_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            he_features = batch["he_features"].to(device)
            he_mask = batch["he_mask"].to(device)
            mif_pos_features = batch["mif_pos_features"].to(device)
            mif_pos_mask = batch["mif_pos_mask"].to(device)

            # 将负样本列表转换为张量并移动到设备上
            mif_neg_features = [neg.to(device) for neg in batch["mif_neg_features"]]
            mif_neg_mask = [neg.to(device) for neg in batch["mif_neg_mask"]]

            start_index = args.start_index
            mif_channel = args.mif_channel

            _, he_embed, cell_logits = he_model(he_features, he_mask)

            # 处理 he_valid
            he_valid_list = []
            for i in range(he_embed.size(0)):
                valid_he = cell_logits[i][he_mask[i].bool()]
                he_valid_list.append(valid_he)
            he_valid = torch.cat(he_valid_list, dim=0)

            # 处理 mif_pos_valid
            mif_pos_valid_list = []
            for i in range(mif_pos_features.size(0)):
                valid_mif = mif_pos_features[i][mif_pos_mask[i].bool()]
                mif_pos_valid_list.append(valid_mif)
            mif_pos_valid = torch.cat(mif_pos_valid_list, dim=0)

            # 处理所有 mif_neg_valid
            mif_neg_valid_list = []
            for neg_feature, neg_mask in zip(mif_neg_features, mif_neg_mask):
                valid_neg = neg_feature[neg_mask.bool()]
                mif_neg_valid_list.append(valid_neg)
            mif_neg_valid = torch.cat(mif_neg_valid_list, dim=0)

            # Pad features if needed
            if mif_pos_valid.size(-1) != he_valid.size(-1):
                mif_pos_valid = F.pad(mif_pos_valid, (0, he_valid.size(-1) - mif_pos_valid.size(-1)))
                mif_neg_valid = F.pad(mif_neg_valid, (0, he_valid.size(-1) - mif_neg_valid.size(-1)))

            min_len = min(he_valid.size(0), mif_pos_valid.size(0), mif_neg_valid.size(0))
            if min_len == 0:
                continue

            he_trunc = he_valid[:min_len]
            mif_pos_trunc = mif_pos_valid[:min_len]
            mif_neg_trunc = mif_neg_valid[:min_len]

            loss_mse = hungary_mse_loss(he_trunc, mif_pos_trunc, start_index, mif_channel)

            he_anchor = he_trunc.mean(dim=0, keepdim=True)
            mif_pos_anchor = mif_pos_trunc.mean(dim=0, keepdim=True)
            mif_neg_anchor = mif_neg_trunc.mean(dim=0, keepdim=True)
            loss_ctr = contrastive_loss(he_anchor, mif_pos_anchor, mif_neg_trunc)

            loss = args.lambda_mse * loss_mse + args.lambda_contrast * loss_ctr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/mse", loss_mse.item() * args.lambda_mse, global_step)
            writer.add_scalar("loss/contrast", loss_ctr.item() * args.lambda_contrast, global_step)

            epoch_loss += loss.item()
            epoch_mse_loss += loss_mse.item()
            epoch_contrast_loss += loss_ctr.item()
            global_step += 1

        # Logging for epoch loss
        avg_total = epoch_loss / len(train_loader)
        avg_mse = epoch_mse_loss / len(train_loader)
        avg_contrast = epoch_contrast_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Total Loss: {avg_total:.6f} | "
            f"Train MSE Loss: {avg_mse:.6f} | "
            f"Train Contrast Loss: {avg_contrast:.6f}"
        )
        recent_model_path = os.path.join(exp_dir, f"he_model_recent.pth")
        torch.save(he_model.state_dict(), recent_model_path)

        # Test evaluation
        he_model.eval()
        test_loss = 0.0
        test_mse_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}/{args.epochs}"):
                he_features = batch["he_features"].to(device)
                he_mask = batch["he_mask"].to(device)
                mif_pos_features = batch["mif_pos_features"].to(device)
                mif_pos_mask = batch["mif_pos_mask"].to(device)

                # Load negative samples
                mif_neg_features = [neg.to(device) for neg in batch["mif_neg_features"]]
                mif_neg_mask = [neg.to(device) for neg in batch["mif_neg_mask"]]

                _, he_embed, cell_logits = he_model(he_features, he_mask)

                # Process he_valid
                he_valid_list = []
                for i in range(he_embed.size(0)):
                    valid_he = cell_logits[i][he_mask[i].bool()]
                    he_valid_list.append(valid_he)
                he_valid = torch.cat(he_valid_list, dim=0)

                # Process mif_pos_valid
                mif_pos_valid_list = []
                for i in range(mif_pos_features.size(0)):
                    valid_mif = mif_pos_features[i][mif_pos_mask[i].bool()]
                    mif_pos_valid_list.append(valid_mif)
                mif_pos_valid = torch.cat(mif_pos_valid_list, dim=0)

                # Process mif_neg_valid
                mif_neg_valid_list = []
                for neg_feature, neg_mask in zip(mif_neg_features, mif_neg_mask):
                    valid_neg = neg_feature[neg_mask.bool()]
                    mif_neg_valid_list.append(valid_neg)
                mif_neg_valid = torch.cat(mif_neg_valid_list, dim=0)

                loss_mse = hungary_mse_loss(he_valid, mif_pos_valid, start_index, mif_channel)
                test_mse_loss += loss_mse.item()

        avg_test_mse = test_mse_loss / len(test_loader)
        writer.add_scalar("test/mse_loss", avg_test_mse, epoch)

        logger.info(f"Test MSE Loss: {avg_test_mse:.6f}")

        # Save the best model based on test MSE
        if avg_test_mse < best_loss:
            best_loss = avg_test_mse
            best_model_path = os.path.join(exp_dir, f"he_model_best.pth")
            torch.save(he_model.state_dict(), best_model_path)
            logger.info(f"New best model saved with loss: {best_loss:.6f}")


    writer.close()
    logger.info("Training finished")


# =========================
# 参数
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--he_model_path", default=None)

    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--mif_channel", type=int, required=True)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_contrast", type=float, default=1.0)

    parser.add_argument("--num_neg_samples", type=int, default=10)  # 负样本数量
    args = parser.parse_args()

    main(args)
