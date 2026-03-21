import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from PIL import Image
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt

# 增加PIL图像大小限制以支持大型SVS文件
Image.MAX_IMAGE_PIXELS = None
import argparse
from collections import defaultdict
from copy import deepcopy
from torchvision import transforms
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import openslide

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import load_cellpose_model, preprocess_image, build_cell_features
from updated_models import TransformerEncoder
from CellEngine import CellInferenceEngine

# Try to import ctranpath, install if not available
try:
    from module.TransPath.ctran import ctranspath
except ImportError:
    raise ImportError("CTransPath module not found. Please install it from https://github.com/Xiyue-Wang/TransPath")


DEFAULT_ENGINE_CONFIG = {
    "cellpose_model_path": "/home/zyh/NewMedLabel/medlabel_image_cell_inference/model/cpsam",
    "ctranspath_checkpoint": "/home/zyh/NewMedLabel/XCellAligner/module/checkpoint/ctranspath.pth",
    "xcell_checkpoint": "/home/zyh/NewMedLabel/XCellAligner/module/checkpoint/he_model_best.pth",
    "xcell_config": {
        "input_dim": 768,
        "hidden_dim": 512,
        "n_heads": 8,
        "num_layers": 4,
        "output_dim": 20,
        "use_large_vit": True,
        "vit_weights_path": "/home/zyh/NewMedLabel/XCellAligner/module/vit/vit-huge-patch14-224-in21k"
    },
    "device": "cuda"
}

_inference_engine = None
_engine_config_snapshot = None
_inference_engine_lock = threading.Lock()
_gpu_lock = threading.Lock()

def _deep_merge_dict(base_dict, override_dict):
    if not override_dict:
        return deepcopy(base_dict)
    merged = deepcopy(base_dict)
    
    for key, value in override_dict.items():
        if (
            isinstance(value, dict)
            and key in merged
            and isinstance(merged[key], dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def get_cell_inference_engine(engine_config=None, force_recreate=False):
    global _inference_engine, _engine_config_snapshot
    with _inference_engine_lock:
        if force_recreate or _inference_engine is None:
            merged_config = _deep_merge_dict(DEFAULT_ENGINE_CONFIG, engine_config or {})
            _engine_config_snapshot = merged_config
            _inference_engine = CellInferenceEngine(**merged_config)
        elif engine_config and engine_config != _engine_config_snapshot:
            print("[Warning] Ignoring new engine_config because engine is already initialized. Use force_recreate=True if needed.")
    return _inference_engine

def save_features_to_disk(features, image_name, output_dir):
    features_file = os.path.join(output_dir, f"features_{image_name}.npy")
    if features is None:
        print(f"[SaveFeatures] 跳过保存（features 为空）: {image_name}")
        return None
    if isinstance(features, np.ndarray):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        num_cells = features.shape[0]
        feature_dim = features.shape[1] if features.ndim > 1 else 1
        print(f"[SaveFeatures] 开始保存: {image_name} | 细胞数={num_cells} 维度={feature_dim}")
        memmap = np.lib.format.open_memmap(
            features_file, mode="w+", dtype=features.dtype, shape=(num_cells, feature_dim)
        )
        chunk_size = 100000
        total_chunks = (num_cells + chunk_size - 1) // chunk_size
        for start in range(0, num_cells, chunk_size):
            end = min(start + chunk_size, num_cells)
            memmap[start:end] = features[start:end]
            chunk_index = (start // chunk_size) + 1
            if total_chunks > 1:
                print(f"[SaveFeatures] 写入进度: {chunk_index}/{total_chunks}")
        del memmap
        print(f"[SaveFeatures] 保存完成: {features_file}")
        return features_file
    if len(features) == 0:
        empty_array = np.empty((0, 0), dtype=np.float32)
        np.save(features_file, empty_array)
        print(f"[SaveFeatures] 保存空特征: {features_file}")
        return features_file
    first = np.asarray(features[0])
    if first.ndim == 0:
        feature_dim = 1
    elif first.ndim == 1:
        feature_dim = first.shape[0]
    else:
        feature_dim = first.shape[1]
    num_cells = len(features)
    print(f"[SaveFeatures] 开始保存: {image_name} | 细胞数={num_cells} 维度={feature_dim}")
    memmap = np.lib.format.open_memmap(
        features_file, mode="w+", dtype=first.dtype, shape=(num_cells, feature_dim)
    )
    chunk_size = 100000
    total_chunks = (num_cells + chunk_size - 1) // chunk_size
    for start in range(0, num_cells, chunk_size):
        end = min(start + chunk_size, num_cells)
        chunk = np.asarray(features[start:end])
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        memmap[start:end] = chunk
        chunk_index = (start // chunk_size) + 1
        if total_chunks > 1:
            print(f"[SaveFeatures] 写入进度: {chunk_index}/{total_chunks}")
    del memmap
    print(f"[SaveFeatures] 保存完成: {features_file}")
    return features_file

def save_masks_to_disk(masks, image_name, output_dir):
    """
    将巨型掩码保存为 NumPy 文件，避免撑爆内存
    """
    masks_file = os.path.join(output_dir, f"masks_{image_name}.npy")
    np.save(masks_file, masks)
    print(f"[SaveMasks] 保存完成: {masks_file}")
    return masks_file

def _process_large_image_by_patches(engine, image_path, tile_size=2048, overlap=0, max_workers=4):
    slide = openslide.OpenSlide(image_path)
    width, height = slide.level_dimensions[0]
    slide.close()
    step_size = tile_size - overlap
    full_mask = np.zeros((height, width), dtype=np.int32)
    all_features = []
    all_patch_features = []
    label_offset = 0
    tiles_x = (width + step_size - 1) // step_size
    tiles_y = (height + step_size - 1) // step_size
    total_tiles = tiles_x * tiles_y
    processed_tiles = 0
    tile_tasks = []
    print(f"[TiledInfer] 图像尺寸: {width}x{height}")
    print(f"[TiledInfer] tile_size={tile_size} overlap={overlap} step={step_size}")
    print(f"[TiledInfer] 总分块数: {tiles_x}x{tiles_y}={total_tiles}, 线程数={max_workers}")
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            tile_tasks.append((x, y, w, h))

    thread_local = threading.local()
    opened_slides = []
    opened_slides_lock = threading.Lock()

    def _get_thread_slide():
        local_slide = getattr(thread_local, "slide", None)
        if local_slide is None:
            local_slide = openslide.OpenSlide(image_path)
            thread_local.slide = local_slide
            with opened_slides_lock:
                opened_slides.append(local_slide)
        return local_slide

    def _run_tile(task):
        x, y, w, h = task
        try:
            local_slide = _get_thread_slide()
            tile_img = local_slide.read_region((x, y), 0, (w, h)).convert("RGB")
            tile_np = np.array(tile_img)
            
            with _gpu_lock:
                result = engine.predict(tile_np)
                
            return x, y, w, h, result, None
        except Exception as e:
            return x, y, w, h, None, str(e)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_tile, task) for task in tile_tasks]
            
            # 使用列表顺序遍历而不是 as_completed，确保按空间顺序追加细胞和 Patch特征
            for future in futures:
                x, y, w, h, result, error_message = future.result()
                processed_tiles += 1
                if processed_tiles % 10 == 0 or processed_tiles == total_tiles:
                    print(f"[TiledInfer] 分块进度: {processed_tiles}/{total_tiles}")
                if error_message is not None:
                    print(f"[TiledInfer] 分块失败: x={x}, y={y}, w={w}, h={h}, err={error_message}")
                    continue
                if result is None or result.get("num_cells", 0) == 0:
                    continue
                tile_features = result.get("cell_logits")
                if tile_features is not None:
                    if isinstance(tile_features, np.ndarray) and tile_features.ndim == 1:
                        tile_features = tile_features.reshape(-1, 1)
                    all_features.append(tile_features)
                
                # Extract patch feature (ViT output)
                tile_patch_feat = result.get("cls_output")
                if tile_patch_feat is not None:
                    all_patch_features.append(tile_patch_feat)
                tile_masks = result.get("masks")
                if tile_masks is None:
                    continue
                tile_masks = np.asarray(tile_masks)
                if tile_masks.ndim != 2:
                    tile_masks = tile_masks.squeeze()
                if tile_masks.size == 0:
                    continue
                if tile_masks.max() > 0:
                    tile_masks = tile_masks.astype(np.int32).copy()
                    tile_masks[tile_masks > 0] += label_offset
                    label_offset = int(tile_masks.max())
                region = full_mask[y:y+h, x:x+w]
                if region.shape != tile_masks.shape:
                    tile_masks = tile_masks[:region.shape[0], :region.shape[1]]
                mask_new = tile_masks > 0
                region_mask = (region == 0) & mask_new
                region[region_mask] = tile_masks[region_mask]
                full_mask[y:y+h, x:x+w] = region
    finally:
        for opened_slide in opened_slides:
            try:
                opened_slide.close()
            except Exception:
                pass
    if len(all_features) == 0:
        print("[TiledInfer] 未检测到细胞")
        return None
    enhanced_features = np.concatenate(all_features, axis=0)
    
    if len(all_patch_features) > 0:
        patch_features = np.concatenate(all_patch_features, axis=0)
    else:
        patch_features = None
        
    print(f"[TiledInfer] 合并完成: 细胞特征数={enhanced_features.shape[0]} 维度={enhanced_features.shape[1]}")
    return full_mask, enhanced_features, patch_features

def extract_cell_features_for_inference(image_path, cellpose_model, ctranspath_model, device):
    img = preprocess_image(image_path)
    masks, flows, styles = cellpose_model.eval(img, diameter=None, channels=[0, 0])
    ctranspath_model.eval() 
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cell_features = []
    unique_masks = np.unique(masks)
    for label in unique_masks:
        if label == 0:
            continue
        cell_mask = masks == label
        cell_img = img * cell_mask[..., None]
        cell_img_pil = Image.fromarray(cell_img)
        cell_img_tensor = preprocess(cell_img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = ctranspath_model(cell_img_tensor)
        cell_features.append(features.squeeze().cpu().numpy())

    return cell_features, masks, img

def get_dominant_class_in_mask(segment_mask, cell_mask):
    masked_values = segment_mask[cell_mask]
    if len(masked_values) == 0:
        return 0
    unique_values, counts = np.unique(masked_values, return_counts=True)
    dominant_class = unique_values[np.argmax(counts)]
    return dominant_class

def get_cell_centroids(masks):
    """
    极速版：计算每个细胞的质心坐标 (专为WSI大图优化，避免 OOM)
    """
    if masks is None:
        return []
    masks = np.asarray(masks)
    if masks.ndim != 2:
        masks = masks.squeeze()
        
    ys, xs = np.nonzero(masks)
    if ys.size == 0:
        return []
        
    labels = masks[ys, xs].astype(np.int64, copy=False)
    max_label = int(labels.max())
    
    # 利用 bincount 极速统计质心，完全规避 for 循环和 np.unique
    counts = np.bincount(labels, minlength=max_label + 1)
    sum_y = np.bincount(labels, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(labels, weights=xs, minlength=max_label + 1)
    
    valid_labels = np.where(counts > 0)[0]
    valid_labels = valid_labels[valid_labels > 0] # 剔除背景(0)
    
    if valid_labels.size == 0:
        return []
        
    centroid_x = (sum_x[valid_labels] / counts[valid_labels]).astype(np.int64)
    centroid_y = (sum_y[valid_labels] / counts[valid_labels]).astype(np.int64)
    
    return list(zip(centroid_x.tolist(), centroid_y.tolist()))

def visualize_clusters(masks, cluster_labels, save_path, k):
    """
    优化版：可视化聚类结果 (降采样 + 高速映射，避免 OOM)
    """
    masks = np.asarray(masks)
    if masks.ndim != 2:
        masks = masks.squeeze()
        
    h, w = masks.shape
    max_visual_side = 4096
    stride = max(1, int(np.ceil(max(h, w) / max_visual_side)))
    
    if stride > 1:
        print(f"[Visualize] 掩码尺寸过大 ({w}x{h})，按步长 {stride} 下采样后绘图")
        
    masks_view = masks[::stride, ::stride]
    masks_view = np.array(masks_view, dtype=np.int32) # 强制转入内存进行高速处理
    
    # 极速映射逻辑
    num_cells = len(cluster_labels)
    max_label_in_view = masks_view.max() if masks_view.size > 0 else 0
    lut_size = max(num_cells + 1, max_label_in_view + 1)
    
    label_to_cluster = np.zeros(lut_size, dtype=np.int16)
    label_to_cluster[1:num_cells + 1] = np.asarray(cluster_labels, dtype=np.int16) + 1
    
    cluster_map = label_to_cluster[masks_view]
    
    colors = plt.cm.get_cmap('tab10', k)
    color_lut = np.zeros((k + 1, 3), dtype=np.uint8)
    for idx in range(1, k + 1):
        color_lut[idx] = (np.array(colors(idx - 1)[:3]) * 255).astype(np.uint8)
        
    result_img = color_lut[cluster_map]
    result_pil = Image.fromarray(result_img)
    result_pil.save(save_path)
    print(f"结果已保存到: {save_path}")

def pad_features_to_max_cells(features, max_cells=255):
    num_cells, feature_dim = features.shape
    padded_features = np.zeros((max_cells, feature_dim))
    padded_features[:num_cells] = features
    mask = np.zeros(max_cells)
    mask[:num_cells] = 1
    return padded_features, mask

def process_single_image(image_path, k=5, segment_mask=None):
    engine = get_cell_inference_engine()

    use_tiling = False
    if isinstance(image_path, str) and os.path.isfile(image_path):
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".svs", ".tif", ".tiff"]:
            use_tiling = True
        else:
            file_size = os.path.getsize(image_path) / (1024 * 1024)
            if file_size > 300:
                use_tiling = True
    print(f"[SingleImage] {os.path.basename(image_path)} | 分块模式={use_tiling}")
    if use_tiling:
        try:
            tile_workers = min(8, max(1, os.cpu_count() or 1))
            tiled_result = _process_large_image_by_patches(engine, image_path, max_workers=tile_workers)
            if tiled_result is None:
                print(f"未检测到细胞: {os.path.basename(image_path)}")
                return False
            masks, enhanced_features, patch_features = tiled_result
            print(f"[SingleImage] 分块推理完成: 细胞数={len(enhanced_features)} 维度={enhanced_features.shape[1]}")
            if segment_mask is not None and masks is not None:
                print("segment_mask 已提供，但当前流程默认直接使用模型输出特征进行聚类。")
            return masks, enhanced_features, patch_features, os.path.basename(image_path), image_path
        except Exception as e:
            print(f"大图分块推理失败，回退到单图推理: {e}")
            
    with _gpu_lock:
        result = engine.predict(image_path)

    if result is None or result.get("cell_logits") is None or result.get("num_cells", 0) == 0:
        print(f"未检测到细胞: {os.path.basename(image_path)}")
        return False

    masks = result.get("masks")
    enhanced_features = result.get("cell_logits")
    patch_features = result.get("cls_output")
    if isinstance(enhanced_features, np.ndarray) and enhanced_features.ndim == 2:
        print(f"[SingleImage] 单图推理完成: 细胞数={enhanced_features.shape[0]} 维度={enhanced_features.shape[1]}")

    if segment_mask is not None and masks is not None:
        print("segment_mask 已提供，但当前流程默认直接使用模型输出特征进行聚类。")

    return masks, enhanced_features, patch_features, os.path.basename(image_path), image_path

def batch_inference(input_folder, output_folder, k=5, max_workers=2, engine_config=None, force_recreate_engine=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 预初始化推理引擎
    get_cell_inference_engine(engine_config=engine_config, force_recreate=force_recreate_engine)

    # 创建输出文件夹与临时文件夹
    os.makedirs(output_folder, exist_ok=True)
    temp_features_dir = os.path.join(output_folder, "temp_features")
    temp_masks_dir = os.path.join(output_folder, "temp_masks")
    os.makedirs(temp_features_dir, exist_ok=True)
    os.makedirs(temp_masks_dir, exist_ok=True)
    
    print(f"[Batch] 输出目录: {output_folder}")
    print(f"[Batch] 临时特征目录: {temp_features_dir}")
    print(f"[Batch] 临时掩码目录: {temp_masks_dir}")
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.svs']
    image_files = []

    if os.path.isfile(input_folder):
        if any(input_folder.lower().endswith(ext[1:]) for ext in image_extensions):
            image_files = [input_folder]
    else:
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, extension)))
            image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

    if not image_files:
        print(f"在 {input_folder} 中没有找到图像文件")
        return
    
    print(f"[Batch] 找到 {len(image_files)} 个图像文件")
    
    processed_images = []
    image_paths = []
    image_mask_files = []  # 存路径，不再存巨型数组
    segment_mask = None
    
    # 使用线程池处理图像
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {}
        for i, image_path in enumerate(image_files):
            print(f"[Batch] [{i+1}/{len(image_files)}] 准备处理图像: {os.path.basename(image_path)}")
            future = executor.submit(process_single_image, image_path, k, segment_mask)
            future_to_image[future] = (image_path, segment_mask)
        
        completed_count = 0
        for future in as_completed(future_to_image):
            image_path, segment_mask = future_to_image[future]
            try:
                result = future.result(timeout=120) 
                if result:
                    masks, enhanced_features, patch_features, image_name, orig_image_path = result
                    
                    # 1. 保存特征到磁盘
                    saved_path = save_features_to_disk(enhanced_features, image_name, temp_features_dir)
                    
                    # 2. 保存掩码到磁盘
                    saved_mask_path = save_masks_to_disk(masks, image_name, temp_masks_dir)
                    
                    # 3. 保存 ViT 高维特征 (patch level features)
                    if patch_features is not None:
                        patch_dir = os.path.join(output_folder, "patch_features")
                        os.makedirs(patch_dir, exist_ok=True)
                        patch_file = os.path.join(patch_dir, f"{os.path.splitext(image_name)[0]}.npy")
                        np.save(patch_file, patch_features)
                    
                    processed_images.append(image_name)
                    image_paths.append(orig_image_path)
                    image_mask_files.append(saved_mask_path)
                    
                    completed_count += 1
                    print(f"[Batch] [{completed_count}/{len(image_files)}] 已保存 {image_name} 的特征及其掩码")
                else:
                    completed_count += 1
                    print(f"[Batch] [{completed_count}/{len(image_files)}] 跳过图像 {os.path.basename(image_path)}")
            except Exception as e:
                completed_count += 1
                print(f"[Batch] [{completed_count}/{len(image_files)}] 处理图像 {os.path.basename(image_path)} 时出错: {str(e)}")
                continue
    
    print("[Batch] 所有图像特征和掩码已保存到磁盘")
    
    print("[Batch] 正在从磁盘加载所有特征...")
    feature_files = [os.path.join(temp_features_dir, f) for f in os.listdir(temp_features_dir) if f.startswith("features_")]
    all_features = []
    print(f"[Batch] 特征文件数量: {len(feature_files)}")

    for feature_file in feature_files:
        batch_features = np.load(feature_file)

        if batch_features.ndim != 2:
            print(f"跳过非二维特征文件: {feature_file}, shape={batch_features.shape}")
            continue

        feature_dim = batch_features.shape[1]

        if feature_dim == 20:
            pad = np.zeros((batch_features.shape[0], 1), dtype=batch_features.dtype)
            batch_features = np.concatenate([batch_features, pad], axis=1)
        elif feature_dim == 9:
            pass
        else:
            print(f"跳过异常维度特征文件: {feature_file}, shape={batch_features.shape}")
            continue

        all_features.append(batch_features)
    
    if len(all_features) == 0:
        print("[Batch] 没有找到任何特征文件")
        return
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"[Batch] 加载所有特征完成，特征总数：{len(all_features)}")
    
    print(f"[Batch] 使用K-means进行 {k} 类聚类...")
    
    # 聚类防死锁 + 大数据量自动切换 MiniBatchKMeans
    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
        if len(all_features) >= 100000:
            print("[Batch] 细胞数量超过10万，自动切换为 MiniBatchKMeans 以加速计算...")
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=8192)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(all_features)
    
    cluster_idx = 0
    all_cell_info = [] 
    
    for i, image_name in enumerate(processed_images):
        try:
            features_file = os.path.join(temp_features_dir, f"features_{image_name}.npy")
            
            if not os.path.exists(features_file):
                print(f"缺少 {image_name} 的特征文件")
                continue
            
            enhanced_features = np.load(features_file)
            base_name = os.path.splitext(image_name)[0]
            
            current_clusters = clusters[cluster_idx:cluster_idx+len(enhanced_features)]
            cluster_idx += len(enhanced_features)
            
            # 从硬盘加载对应这张图的 mask (使用内存映射 mmap_mode 避免 OOM)
            masks_file = image_mask_files[i]
            if not os.path.exists(masks_file):
                print(f"缺少 {image_name} 的掩码文件")
                continue
                
            masks = np.load(masks_file, mmap_mode='r')
            
            # 计算每个细胞的质心坐标 (已替换为极速计算法)
            centroids = get_cell_centroids(masks)
            
            for j, (centroid, cluster_label) in enumerate(zip(centroids, current_clusters)):
                cell_info = {
                    "image_name": image_name,
                    "cell_id": j,
                    "centroid_x": int(centroid[0]),
                    "centroid_y": int(centroid[1]),
                    "cluster": int(cluster_label)
                }
                all_cell_info.append(cell_info)
            
            save_path = os.path.join(output_folder, f"{base_name}_cluster.png")
            
            print(f"可视化聚类结果: {image_name}...")
            # 渲染图像 (已替换为极速降采样法)
            visualize_clusters(masks, current_clusters, save_path, k)
            
            # 极其重要：处理完一张图后，立刻释放它的 mask 对象
            del masks
            
        except Exception as e:
            print(f"处理图像 {image_name} 的聚类结果时出错: {str(e)}")
            continue
    
    all_cell_info_path = os.path.join(output_folder, "all_cells_info.json")
    with open(all_cell_info_path, "w", encoding="utf-8") as f:
        json.dump(all_cell_info, f, ensure_ascii=False, indent=2)
    print(f"所有细胞信息已保存到: {all_cell_info_path}")
    
    print("批量推理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理细胞图像")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output_folder", type=str, required=True, help="结果保存文件夹路径")
    parser.add_argument("--k", type=int, default=5, help="聚类数")
    parser.add_argument("--max_workers", type=int, default=2, help="最大线程数")
    args = parser.parse_args()
    batch_inference(args.input_folder, args.output_folder, args.k, args.max_workers)