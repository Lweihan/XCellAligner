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
# from updated_models import TransformerEncoder


# =========================
# 日志
# =========================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("TRAIN")


# =========================
# utils
# =========================
def parse_xy(filename):
    m = re.search(r"x(\d+)_y(\d+)", filename)
    if m is None:
        raise ValueError(f"非法文件名: {filename}")
    return int(m.group(1)), int(m.group(2))


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def hungary_mse_loss(he_feat, mif_feat, start_index, mif_channel):
    he_slice = he_feat[:, start_index : start_index + mif_channel]
    mif_slice = mif_feat[:, :mif_channel]
    return F.mse_loss(he_slice, mif_slice)


def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos = (anchor * positive).sum(dim=-1, keepdim=True)
    neg = (anchor.unsqueeze(1) * negatives).sum(dim=-1)

    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


# =========================
# Dataset
# =========================
class HeMifDataset(Dataset):
    def __init__(
        self,
        he_dir,
        mif_dir,
        task_id,
        start_index,
        mif_channel,
        num_neg_samples=3,
    ):
        self.task_id = task_id
        self.start_index = start_index
        self.mif_channel = mif_channel
        self.num_neg_samples = num_neg_samples

        self.he_files = [f for f in os.listdir(he_dir) if f.endswith(".pkl")]
        self.he_dir = he_dir
        self.mif_dir = mif_dir

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

    def __len__(self):
        return len(self.valid_pairs)

    def _get_farthest_negative_samples(self, x, y):
        others = [c for c in self.all_coords if c != (x, y)]
        if len(others) <= self.num_neg_samples:
            return random.sample(others, len(others))
        return random.sample(others, self.num_neg_samples)

    def __getitem__(self, idx):
        he_file, (x, y) = self.valid_pairs[idx]

        he = load_pkl(os.path.join(self.he_dir, he_file))
        mif_pos = load_pkl(self.mif_map[(x, y)])

        neg_coords = self._get_farthest_negative_samples(x, y)
        mif_neg = [load_pkl(self.mif_map[c]) for c in neg_coords]

        return {
            "he_features": he["features"][0],
            "he_mask": he["mask"][0],
            "mif_pos_features": mif_pos["features"][0],
            "mif_pos_mask": mif_pos["mask"][0],
            "mif_neg_features": [n["features"][0] for n in mif_neg],
            "mif_neg_mask": [n["mask"][0] for n in mif_neg],
            # ⭐ task meta
            "task_id": self.task_id,
            "start_index": self.start_index,
            "mif_channel": self.mif_channel,
        }


# =========================
# main
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(os.path.join(exp_dir, "logs"))
    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))

    assert (
        len(args.cache_dir)
        == len(args.start_index)
        == len(args.mif_channel)
    )

    num_tasks = len(args.cache_dir)

    train_loaders, test_loaders = [], []

    for t in range(num_tasks):
        dataset = HeMifDataset(
            he_dir=os.path.join(args.cache_dir[t], "he"),
            mif_dir=os.path.join(args.cache_dir[t], "mif"),
            task_id=t,
            start_index=args.start_index[t],
            mif_channel=args.mif_channel[t],
            num_neg_samples=args.num_neg_samples,
        )

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        train_loaders.append(
            DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        )
        test_loaders.append(
            DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
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

    optimizer = AdamW(he_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        he_model.train()

        for task_id, loader in enumerate(train_loaders):
            for batch in tqdm(loader, desc=f"Epoch {epoch+1} | Task {task_id}"):
                start_index = batch["start_index"][0]
                mif_channel = batch["mif_channel"][0]

                he_feat = batch["he_features"].to(device)
                he_mask = batch["he_mask"].to(device)
                mif_pos = batch["mif_pos_features"].to(device)
                mif_pos_mask = batch["mif_pos_mask"].to(device)

                mif_neg = [n.to(device) for n in batch["mif_neg_features"]]
                mif_neg_mask = [n.to(device) for n in batch["mif_neg_mask"]]

                _, _, cell_logits, _ = he_model(raw_images=None, x=he_feat, mask=he_mask)

                he_valid = torch.cat(
                    [cell_logits[i][he_mask[i].bool()] for i in range(he_feat.size(0))]
                )
                mif_pos_valid = torch.cat(
                    [mif_pos[i][mif_pos_mask[i].bool()] for i in range(mif_pos.size(0))]
                )
                mif_neg_valid_list = []

                for neg_feat, neg_mask in zip(mif_neg, mif_neg_mask):
                    # neg_feat: [B, max_cells, C]
                    # neg_mask: [B, max_cells]
                    for i in range(neg_feat.size(0)):
                        valid_neg = neg_feat[i][neg_mask[i].bool()]
                        if valid_neg.numel() > 0:
                            mif_neg_valid_list.append(valid_neg)

                if len(mif_neg_valid_list) == 0:
                    continue

                mif_neg_valid = torch.cat(mif_neg_valid_list, dim=0)

                min_len = min(
                    he_valid.size(0), mif_pos_valid.size(0), mif_neg_valid.size(0)
                )
                if min_len == 0:
                    continue
                
                he_valid_slice = he_valid[
                    :, start_index : start_index + mif_channel
                ]
                he_valid = he_valid[:min_len]
                mif_pos_valid = mif_pos_valid[:min_len]
                mif_neg_valid = mif_neg_valid[:min_len]

                loss_mse = hungary_mse_loss(
                    he_valid, mif_pos_valid, start_index, mif_channel
                )

                he_anchor = he_valid_slice.mean(dim=0, keepdim=True)
                mif_pos_anchor = mif_pos_valid.mean(dim=0, keepdim=True)

                loss_ctr = contrastive_loss(
                    he_anchor,
                    mif_pos_anchor,
                    mif_neg_valid,   # [N_neg, mif_channel]
                )

                loss = args.lambda_mse * loss_mse + args.lambda_contrast * loss_ctr

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar(f"task_{task_id}/mse", loss_mse.item(), global_step)
                writer.add_scalar(f"task_{task_id}/ctr", loss_ctr.item(), global_step)
                global_step += 1
                torch.save(he_model.state_dict(), os.path.join(exp_dir, "he_model_recent.pth"))

        logger.info(f"Epoch {epoch+1} finished: MSE={loss_mse.item():.4f}, Contrastive={loss_ctr.item():.4f}")
        if loss.item() < best_loss:
            print("New best model found! Saving...")
            best_loss = loss.item()    
            torch.save(he_model.state_dict(), os.path.join(exp_dir, "he_model_best.pth"))
    writer.close()
    logger.info("Training done")


# =========================
# args
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", nargs="+", required=True)
    parser.add_argument("--start_index", nargs="+", type=int, required=True)
    parser.add_argument("--mif_channel", nargs="+", type=int, required=True)

    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_contrast", type=float, default=1.0)
    parser.add_argument("--num_neg_samples", type=int, default=10)

    args = parser.parse_args()
    main(args)
