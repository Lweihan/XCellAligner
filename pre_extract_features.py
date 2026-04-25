import os
import re
import sys
import torch
import pickle
import argparse
import hashlib
import logging
from skimage import io
import torch.nn.functional as F
from torchvision import transforms
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="channels deprecated in v4.0.1+.*")

# =========================
# 外部模块
# =========================
from utils import load_cellpose_model
from module.TransPath.ctran import ctranspath
from module.ModalEncoder.cell_density_extractor import CellDensityExtractor


# =========================
# Logging
# =========================
def setup_logger(log_dir, gpu_id=None):
    os.makedirs(log_dir, exist_ok=True)

    tag = f"gpu{gpu_id}" if gpu_id is not None else "main"
    log_file = os.path.join(log_dir, f"pre_extract_{tag}.log")

    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][PID:%(process)d][GPU:%(name)s] %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# =========================
# Filename utils
# =========================
def parse_filename(filename):
    he = re.match(r"he_x(\d+)_y(\d+)\.", filename)
    if he:
        return "he", int(he.group(1)), int(he.group(2))

    mif = re.match(r"mF\d+_x(\d+)_y(\d+)\.", filename)
    if mif:
        return "mif", int(mif.group(1)), int(mif.group(2))

    return None, None, None


def he_cache_path(cache_dir, he_path):
    name = os.path.basename(he_path)
    _, x, y = parse_filename(name)
    if x is not None:
        return os.path.join(cache_dir, "he", f"he_feature_x{x}_y{y}.pkl")
    h = hashlib.md5(he_path.encode()).hexdigest()
    return os.path.join(cache_dir, "he", f"he_feature_{h}.pkl")


def mif_cache_path(cache_dir, mif_paths):
    name = os.path.basename(mif_paths[0])
    _, x, y = parse_filename(name)
    if x is not None:
        return os.path.join(cache_dir, "mif", f"mif_feature_x{x}_y{y}.pkl")
    h = hashlib.md5("_".join(sorted(mif_paths)).encode()).hexdigest()
    return os.path.join(cache_dir, "mif", f"mif_feature_{h}.pkl")


# =========================
# HE feature extraction
# =========================
# def extract_he_feature(
#     he_path, cache_dir, device, cellpose_model, ctp_model, logger
# ):
#     cache_file = he_cache_path(cache_dir, he_path)
#     if os.path.exists(cache_file):
#         logger.info(f"[SKIP] HE cache exists: {os.path.basename(cache_file)}")
#         return

#     try:
#         img = np.array(Image.open(he_path).convert("RGB"))
#         masks, _, _ = cellpose_model.eval(img, diameter=18, channels=[0, 1, 2])

#         from torchvision import transforms

#         preprocess = transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225],
#                 ),
#             ]
#         )

#         feats = []
#         for lab in np.unique(masks):
#             if lab == 0:
#                 continue
#             roi = img * (masks == lab)[..., None]
#             roi = preprocess(Image.fromarray(roi)).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 f = ctp_model(roi)
#             feats.append(f.squeeze().cpu().numpy())

#         max_cells = 255
#         feat_dim = 1000
#         feat_arr = np.zeros((max_cells, feat_dim), dtype=np.float32)
#         mask_arr = np.zeros(max_cells, dtype=np.float32)

#         for i, f in enumerate(feats[:max_cells]):
#             feat_arr[i] = f
#             mask_arr[i] = 1

#         with open(cache_file, "wb") as f:
#             pickle.dump(
#                 {
#                     "features": torch.from_numpy(feat_arr).unsqueeze(0),
#                     "mask": torch.from_numpy(mask_arr).unsqueeze(0),
#                     "cell_masks": masks,
#                 },
#                 f,
#             )

#         logger.info(
#             f"[OK] HE features extracted: {os.path.basename(he_path)} "
#             f"(cells={int(mask_arr.sum())})"
#         )

#     except Exception as e:
#         logger.exception(f"[FAIL] HE extraction failed: {he_path} | {e}")

def extract_he_feature(he_path, cache_dir, device, cellpose_model, ctp_model, logger, masks=None):
    """
    提取 HE 特征。
    
    参数:
        masks (np.ndarray, optional): 预提供的细胞分割掩码。如果提供，将跳过 CellPose 分割步骤。
                                      形状应为 (H, W)， dtype 为 int 或 uint8。
    """
    cache_file = he_cache_path(cache_dir, he_path) # 假设 he_cache_path 已在外部定义
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        features = data['features'][:, 0, :]
        if features.shape[-1] == 768:
            logger.info(f"[SKIP] HE cache exists: {os.path.basename(cache_file)}")
            return

    try:
        # 1. 读取图像
        try:
            img_np = io.imread(he_path)
        except:
            img_np = np.array(Image.open(he_path).convert("RGB"))

        # 2. 图像格式标准化 (确保为 RGB uint8)
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.ndim == 3:
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            elif img_np.shape[2] >= 4:
                img_np = img_np[:, :, :3]

        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        # 3. 获取 Masks (核心修改部分)
        if masks is not None:
            logger.info(f"[INFO] Using provided masks, skipping CellPose segmentation.")
            # 确保 masks 是 numpy 数组以便后续处理统一
            if isinstance(masks, torch.Tensor):
                masks_np = masks.cpu().numpy()
            else:
                masks_np = masks
            
            # 可选：验证 masks 形状是否与图像匹配
            if masks_np.shape[:2] != img_np.shape[:2]:
                logger.warning(f"[WARN] Provided masks shape {masks_np.shape} does not match image shape {img_np.shape}. Resizing or cropping might be needed, but proceeding as is.")
            
            masks = masks_np
        else:
            # 原有逻辑：使用 CellPose 进行分割
            img_for_cellpose = img_np
            masks, _, _ = cellpose_model.eval(img_for_cellpose, diameter=18, channels=[0, 0])

        # 4. 后续处理 (边缘过滤，然后将 masks 转为 Tensor 并提取特征)
        # 注意：此时 masks 变量保证是 numpy 数组
        margin = 20
        h, w = img_np.shape[:2]
        filtered_masks = np.zeros_like(masks)
        for label in np.unique(masks):
            if label == 0:
                continue
            coords = np.where(masks == label)
            if len(coords[0]) == 0:
                continue
            center_y = int(np.mean(coords[0]))
            center_x = int(np.mean(coords[1]))
            # 如果细胞中心距离边界小于 margin，则保留，否则丢弃
            if margin <= center_x < w - margin and margin <= center_y < h - margin:
                filtered_masks[masks == label] = label
        
        masks = filtered_masks  # 更新为过滤后的掩码，确保后续 mIF 也使用相同的 mask
        masks_tensor = torch.from_numpy(masks).to(device).to(torch.int32)
        
        ctp_model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        cell_features = []
        unique_labels = torch.unique(masks_tensor)
        unique_labels = unique_labels[unique_labels != 0]

        for label in unique_labels:
            cell_mask = (masks_tensor == label).float()
            # 使用 numpy 进行掩码操作
            cell_region_np = (img_np * cell_mask.cpu().numpy().astype(np.uint8)[:, :, None])
            
            if cell_region_np.sum() == 0:
                continue
                
            cell_img_pil = Image.fromarray(cell_region_np)
            
            try:
                input_tensor = preprocess(cell_img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = ctp_model(input_tensor)
                
                if output.dim() == 4 and output.shape[-1] == 768:
                    output_permuted = output.permute(0, 3, 1, 2) # [1, 768, 7, 7]
                    feat_pooled = F.adaptive_avg_pool2d(output_permuted, (1, 1)) #[1, 768, 1, 1]
                    cell_img_feat = feat_pooled.squeeze(-1).squeeze(-1) # [1, 768]
                else:
                    print(f"[Error] CTranspath extract feature failed.")
                    continue

                cell_features.append(cell_img_feat.squeeze(0).cpu().numpy())
            except Exception as e:
                # 记录单个细胞处理失败，但不中断整体流程
                continue

        if not cell_features:
            logger.warning(f"[WARN] No valid cell features extracted for {he_path}")
            # 如果没有提取到任何特征，可以选择返回或保存空数据，这里选择保存空数据以保持流程一致
            max_cells_limit = 255
            feat_dim = 768 # 默认维度，或者根据模型定义
        else:
            max_cells_limit = 255
            feat_dim = len(cell_features[0])
        
        feat_arr = np.zeros((max_cells_limit, feat_dim), dtype=np.float32)
        mask_arr = np.zeros(max_cells_limit, dtype=np.float32)

        for i, f in enumerate(cell_features[:max_cells_limit]):
            feat_arr[i] = f
            mask_arr[i] = 1.0

        save_data = {
            "features": torch.from_numpy(feat_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "cell_masks": masks # 保存原始的 numpy masks
        }

        with open(cache_file, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"[OK] HE features extracted: {os.path.basename(he_path)} (cells={int(mask_arr.sum())})")

    except Exception as e:
        logger.exception(f"[FAIL] HE extraction failed: {he_path} | {e}")


# =========================
# mIF feature extraction
# =========================
def extract_mif_feature(
    mif_paths, cache_dir, cell_masks, extractor, logger
):
    cache_file = mif_cache_path(cache_dir, mif_paths)
    if os.path.exists(cache_file):
        logger.info(f"[SKIP] mIF cache exists: {os.path.basename(cache_file)}")
        return

    try:
        imgs = [np.array(Image.open(p)) for p in mif_paths]
        density = extractor.process_image_pair(
            imgs, [0] + [1] * (len(imgs) - 1), cell_masks
        )

        max_cells = 255
        feat_dim = density.shape[1]
        feat_arr = np.zeros((max_cells, feat_dim), dtype=np.float32)
        mask_arr = np.zeros(max_cells, dtype=np.float32)

        n = min(max_cells, density.shape[0])
        feat_arr[:n] = density[:n]
        mask_arr[:n] = 1

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "features": torch.from_numpy(feat_arr).unsqueeze(0),
                    "mask": torch.from_numpy(mask_arr).unsqueeze(0),
                },
                f,
            )

        logger.info(
            f"[OK] mIF features extracted: "
            f"{os.path.basename(mif_paths[0])} (cells={n})"
        )

    except Exception as e:
        logger.exception(f"[FAIL] mIF extraction failed: {mif_paths} | {e}")


# =========================
# GPU worker
# =========================
def gpu_worker(
    xgpu_id,
    gpu_id,
    he_list,
    mif_groups,
    cache_dir,
    log_dir,
    progress_q,
):
    gpu_id = int(gpu_id)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    logger = setup_logger(log_dir, xgpu_id)
    logger.info(f"GPU worker started on cuda:{gpu_id}")
    logger.info(f"Setting device to: {gpu_id}, type: {type(gpu_id)}")
    current = torch.cuda.current_device()
    logger.info(f"Current device after set_device: {current}")
    logger.info(f"Assigned HE patches: {len(he_list)}")

    cellpose_model = load_cellpose_model(device=device)
    ctp_model = ctranspath().to(device)
    
    # 强制加载预训练权重
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'module', 'checkpoint', 'ctranspath.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CTransPath weights not found at {checkpoint_path}. You must provide correct weights!")
    logger.info(f"Loading CTransPath weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        ctp_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        ctp_model.load_state_dict(checkpoint, strict=False)

    if hasattr(ctp_model, 'head'):
        ctp_model.head = torch.nn.Identity()
    ctp_model.eval()
    extractor = CellDensityExtractor()

    for he_path, mif_paths in zip(he_list, mif_groups):
        he_cache_file = he_cache_path(cache_dir, he_path)
        if os.path.exists(he_cache_file):
            logger.info(f"[LOAD] Loading existing HE features: {os.path.basename(he_cache_file)}")
            
            # 直接从缓存加载cell_masks
            with open(he_cache_file, "rb") as f:
                cell_masks = pickle.load(f)["cell_masks"]

            extract_he_feature(
                he_path, cache_dir, device, cellpose_model, ctp_model, logger, cell_masks
            )
                
            # 提取mIF特征
            extract_mif_feature(
                mif_paths, cache_dir, cell_masks, extractor, logger
            )
        else:
            extract_he_feature(
                he_path, cache_dir, device, cellpose_model, ctp_model, logger
            )

            with open(he_cache_path(cache_dir, he_path), "rb") as f:
                cell_masks = pickle.load(f)["cell_masks"]

            extract_mif_feature(
                mif_paths, cache_dir, cell_masks, extractor, logger
            )

        progress_q.put(1)

    logger.info("GPU worker finished.")


# =========================
# Main
# =========================
def main(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(args.log_dir)
    logger.info("==== Pre-extract features started ====")

    he_paths = sorted(
        [
            os.path.join(args.he_dir, f)
            for f in os.listdir(args.he_dir)
            if f.startswith("he_x")
        ]
    )

    mif_channels = sorted(os.listdir(args.mif_dir))
    mif_groups = []

    for he in he_paths:
        _, x, y = parse_filename(os.path.basename(he))
        group = [
            os.path.join(
                args.mif_dir, ch, f"mF{ch.split('-')[0]}_x{x}_y{y}.png"
            )
            for ch in mif_channels
        ]
        mif_groups.append(group)

    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        raise RuntimeError("No CUDA devices available")

    logger.info(f"Detected {ngpu} GPUs")
    xgpu = 1 * ngpu

    he_chunks = [he_paths[i::xgpu] for i in range(xgpu)]
    mif_chunks = [mif_groups[i::xgpu] for i in range(xgpu)]

    manager = mp.Manager()
    q = manager.Queue()

    procs = []
    for xgid in range(xgpu):
        gid = xgid % ngpu
        p = mp.Process(
            target=gpu_worker,
            args=(
                xgid,
                gid,
                he_chunks[xgid],
                mif_chunks[xgid],
                args.cache_dir,
                args.log_dir,
                q,
            ),
        )
        p.start()
        procs.append(p)

    with tqdm(total=len(he_paths), desc="Pre-extract (Multi-GPU)") as bar:
        done = 0
        while done < len(he_paths):
            q.get()
            done += 1
            bar.update(1)

    for p in procs:
        p.join()

    logger.info("==== Pre-extract features finished ====")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser("Multi-GPU pre feature extraction")
    parser.add_argument("--he_dir", required=True)
    parser.add_argument("--mif_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--log_dir", default="./logs")

    args = parser.parse_args()
    main(args)
