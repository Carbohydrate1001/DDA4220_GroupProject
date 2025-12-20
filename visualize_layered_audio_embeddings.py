import os
import json
import torch
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import librosa
from tqdm import tqdm
from collections import defaultdict
from modelscope import ClapModel, ClapProcessor
from hyperbolic_projection import HyperbolicProjection

## ==========================================
## 1. Helper Functions
## ==========================================

def flatten_audio_tree(node, name, root_name, depth=0, parent=None, result=None):
    if result is None: result = []
    result.append({'name': name, 'depth': depth, 'parent': parent, 'root_name': root_name})
    for child_name, child_content in node.items():
        flatten_audio_tree(child_content, child_name, root_name, depth + 1, name, result)
    return result

def poincare_distance(u, v):
    sq_dist = np.sum((u - v) ** 2)
    u_norm_sq = np.clip(np.sum(u ** 2), 0, 1 - 1e-6)
    v_norm_sq = np.clip(np.sum(v ** 2), 0, 1 - 1e-6)
    dist = 1 + 2 * sq_dist / ((1 - u_norm_sq) * (1 - v_norm_sq))
    return np.arccosh(np.maximum(dist, 1.0))

def process_audio_from_bytes(audio_bytes, target_sr=48000):
    try:
        with io.BytesIO(audio_bytes) as f:
            audio, sr = librosa.load(f, sr=None)
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        max_val = np.abs(audio).max()
        if max_val > 0: audio = audio / max_val
        return audio.astype(np.float32)
    except Exception:
        return None

## ==========================================
## 2. Visualization Logic
## ==========================================

def visualize_avg_hierarchy(embeddings, nodes_info, output_dir="./Figures"):
    print(f"Generating balanced visualization for {len(embeddings)} averaged labels...")
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.3, metric=poincare_distance, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)

    # 径向平衡
    coords_2d -= np.mean(coords_2d, axis=0)
    norms = np.linalg.norm(coords_2d, axis=1)
    r = np.clip(norms / (np.percentile(norms, 99) + 1e-7), 0, 1)
    new_r = np.power(r, 0.6)
    theta = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    coords_2d[:, 0] = new_r * np.cos(theta)
    coords_2d[:, 1] = new_r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    ax.add_artist(plt.Circle((0, 0), 1.0, color='#E0E0E0', fill=False, lw=2))

    name_to_idx = {node['name']: i for i, node in enumerate(nodes_info)}
    for i, node in enumerate(nodes_info):
        if node['parent'] in name_to_idx:
            p_idx = name_to_idx[node['parent']]
            ax.plot([coords_2d[i, 0], coords_2d[p_idx, 0]], [coords_2d[i, 1], coords_2d[p_idx, 1]], 
                    color='gray', alpha=0.15, lw=0.5)

    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=[n['depth'] for n in nodes_info], 
                        cmap='plasma', s=45, alpha=0.8, edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Hierarchy Depth', fontsize=12)

    # 仅标记根节点或特定节点
    for i, node in enumerate(nodes_info):
        if node['depth'] == 0:
            ax.text(coords_2d[i, 0], coords_2d[i, 1], node['name'], fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.axis('off')
    plt.title("Poincaré Disk: Averaged Audio Embeddings (Filtered Trees < 5 nodes)")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "avg_audio_hierarchy_filtered.png"), dpi=300)
    plt.show()

# --- 新增的聚类图 (By Root Tree) ---
def visualize_audio_by_cluster(embeddings, nodes_info, output_dir="./Figures"):
    print("Generating cluster-based visualization (by tree root)...")
    
    # 使用相同的投影逻辑确保坐标一致性
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.3, metric=poincare_distance, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)

    # 径向平衡
    coords_2d -= np.mean(coords_2d, axis=0)
    norms = np.linalg.norm(coords_2d, axis=1)
    r = np.clip(norms / (np.percentile(norms, 99) + 1e-7), 0, 1)
    new_r = np.power(r, 0.6)
    theta = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    coords_2d[:, 0] = new_r * np.cos(theta)
    coords_2d[:, 1] = new_r * np.sin(theta)

    # 定义颜色映射 (同 text 脚本)
    custom_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf']
    root_names = [n['root_name'] for n in nodes_info]
    unique_roots = list(dict.fromkeys(root_names)) 
    root_to_color = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(unique_roots)}
    colors = [root_to_color[n['root_name']] for n in nodes_info]

    fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    ax.add_artist(plt.Circle((0, 0), 1.0, color='#E0E0E0', fill=False, lw=2, zorder=0))

    name_to_idx = {node['name']: i for i, node in enumerate(nodes_info)}
    for i, node in enumerate(nodes_info):
        if node['parent'] in name_to_idx:
            p_idx = name_to_idx[node['parent']]
            ax.plot([coords_2d[i, 0], coords_2d[p_idx, 0]], [coords_2d[i, 1], coords_2d[p_idx, 1]], 
                    color='gray', alpha=0.08, lw=0.5, zorder=1)

    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=55, alpha=0.9, edgecolors='white', linewidths=0.3, zorder=2)

    for i, node in enumerate(nodes_info):
        if node['depth'] == 0:
            root_col = root_to_color[node['name']]
            ax.text(coords_2d[i, 0], coords_2d[i, 1], node['name'], fontsize=12, fontweight='bold', ha='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=root_col, boxstyle='round,pad=0.3', lw=2))

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect('equal'); ax.axis('off')
    plt.title("Poincaré Disk: Audio Embeddings Clustered by Roots", fontsize=16)
    plt.savefig(os.path.join(output_dir, "avg_audio_cluster_filtered.png"), dpi=300, bbox_inches='tight')
    plt.show()

## ==========================================
## 3. Main Execution
## ==========================================

if __name__ == "__main__":
    PARQUET_DIR = r"E:\Desktop\DDA4220\DDA4220_GroupProject\Datasets\AudioSet\eval"
    TREE_PATH = r"E:\Desktop\DDA4220\DDA4220_GroupProject\Datasets\AudioSet\tree.json"
    PROJ_CHECKPOINT = r"E:\Desktop\DDA4220\DDA4220_GroupProject\checkpoints_hyperbolic_c0_006_t1\best_projection_epoch_10.pth"
    CLAP_PATH = r"C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = ClapModel.from_pretrained(CLAP_PATH).to(device)
    processor = ClapProcessor.from_pretrained(CLAP_PATH)
    ckpt = torch.load(PROJ_CHECKPOINT, map_location=device)
    projection = HyperbolicProjection(input_dim=512, output_dim=512, c=ckpt.get('c', 1.0)).to(device)
    projection.load_state_dict(ckpt['projection_state_dict'])
    model.eval(); projection.eval()

    # 1. 提取所有音频嵌入并按标签累积（支持多标签分发）
    label_to_embeddings = defaultdict(list)
    parquet_files = [os.path.join(PARQUET_DIR, f) for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')]
    
    print("Step 1: Extracting Audio Embeddings from Parquet...")
    for p_file in parquet_files:
        df = pd.read_parquet(p_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(p_file)}"):
            audio_data = process_audio_from_bytes(row['audio']['bytes'])
            raw_labels = row.get('human_labels', row.get('label', 'unknown'))
            
            # 多标签处理：统一转为列表
            if isinstance(raw_labels, str):
                labels = [l.strip(" '[]") for l in raw_labels.split(',')]
            elif isinstance(raw_labels, (list, np.ndarray)):
                labels = [str(l) for l in raw_labels]
            else:
                labels = [str(raw_labels)]
            
            if audio_data is not None:
                with torch.no_grad():
                    inputs = processor(audios=audio_data, sampling_rate=48000, return_tensors="pt").to(device)
                    embed = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
                    hyp_embed = projection(embed).cpu().numpy()
                    for label in labels:
                        label_to_embeddings[label].append(hyp_embed)

    # 计算标签平均值
    avg_label_map = {l: np.mean(np.vstack(es), axis=0) for l, es in label_to_embeddings.items() if es}

    # 2. 读取树结构并过滤掉有效节点数 < 5 的子树
    with open(TREE_PATH, 'r', encoding='utf-8') as f:
        full_tree = json.load(f)

    final_nodes = []
    final_embeddings = []

    print("Step 2: Filtering trees with fewer than 5 valid nodes...")
    for root_key, root_content in full_tree.items():
        # 获取该子树的所有节点
        sub_nodes = flatten_audio_tree(root_content, root_key, root_name=root_key)
        
        # 找出该子树中在 Parquet 数据中实际存在的节点
        valid_sub_nodes = [n for n in sub_nodes if n['name'] in avg_label_map]
        
        # 核心过滤逻辑：如果该子树有效节点数 < 5，跳过整棵树
        if len(valid_sub_nodes) < 5:
            print(f"Skipping tree '{root_key}': only {len(valid_sub_nodes)} valid nodes found.")
            continue
        
        # 如果通过过滤，添加到最终列表
        for node in valid_sub_nodes:
            final_nodes.append(node)
            final_embeddings.append(avg_label_map[node['name']])

    # 3. 绘图
    if final_embeddings:
        all_emb_matrix = np.vstack(final_embeddings)
        # 第一张图：按 Depth 着色
        visualize_avg_hierarchy(all_emb_matrix, final_nodes)
        # 第二张图：按 Tree Root 着色 (聚类)
        visualize_audio_by_cluster(all_emb_matrix, final_nodes)
    else:
        print("No trees met the requirement (nodes >= 5).")