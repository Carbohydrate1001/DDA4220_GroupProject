import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from modelscope import ClapModel, ClapProcessor
from hyperbolic_projection import HyperbolicProjection

## ==========================================
## 1. Helper Functions
## ==========================================

def flatten_audio_tree(node, name, root_name, depth=0, parent=None, result=None):
    if result is None: result = []
    # 确保每个节点都记录它属于哪个根节点 (root_name)
    result.append({
        'name': name, 
        'depth': depth, 
        'parent': parent, 
        'root_name': root_name
    })
    for child_name, child_content in node.items():
        # 注意这里：必须把 root_name 传下去
        flatten_audio_tree(child_content, child_name, root_name, depth + 1, name, result)
    return result

def poincare_distance(u, v):
    sq_dist = np.sum((u - v) ** 2)
    u_norm_sq = np.sum(u ** 2)
    v_norm_sq = np.sum(v ** 2)
    u_norm_sq = min(u_norm_sq, 1 - 1e-6)
    v_norm_sq = min(v_norm_sq, 1 - 1e-6)
    dist = 1 + 2 * sq_dist / ((1 - u_norm_sq) * (1 - v_norm_sq))
    return np.arccosh(np.maximum(dist, 1.0))

## ==========================================
## 2. Updated Combined Visualization
## ==========================================

def visualize_combined_trees(embeddings, all_nodes_info, output_dir="./Figures"):
    print("Re-projecting with Radial Balancing...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 降低 n_neighbors 让算法“多看局部，少看全局”
    # 这样可以防止不同树之间巨大的距离把树内部的层级距离挤掉
    reducer = umap.UMAP(
        n_neighbors=15,      # 减小这个值，找回树内部的结构
        min_dist=0.3,        # 适中的间距
        metric=poincare_distance, 
        random_state=42,
        output_metric='euclidean' 
    )
    
    coords_2d = reducer.fit_transform(embeddings)

    # 2. 中心化与初步缩放
    coords_2d -= np.mean(coords_2d, axis=0)
    norms = np.linalg.norm(coords_2d, axis=1)
    
    # 3. 核心步骤：径向幂次变换 (Radial Power Transform)
    # 通过对半径取 0.5 次方（开方），强制将聚集在中心附近的点向边缘推
    # 这种方法能有效填补圆盘边缘的空白，让分布变均匀
    r = norms / (np.percentile(norms, 99) + 1e-7) # 归一化半径
    r = np.clip(r, 0, 1)
    new_r = np.power(r, 0.6) # 幂次越小，点越往边缘推（0.5-0.7 比较理想）
    
    # 重新计算坐标
    theta = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    coords_2d[:, 0] = new_r * np.cos(theta)
    coords_2d[:, 1] = new_r * np.sin(theta)

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    circle = plt.Circle((0, 0), 1.0, color='#E0E0E0', fill=False, lw=2, zorder=0)
    ax.add_artist(circle)

    name_to_idx = {node['name']: i for i, node in enumerate(all_nodes_info)}
    depths = np.array([n['depth'] for n in all_nodes_info])
    
    # 绘制连线：降低透明度，避免遮盖颜色
    for i, node in enumerate(all_nodes_info):
        if node['parent'] in name_to_idx:
            p_idx = name_to_idx[node['parent']]
            ax.plot([coords_2d[i, 0], coords_2d[p_idx, 0]], 
                    [coords_2d[i, 1], coords_2d[p_idx, 1]], 
                    color='gray', alpha=0.1, lw=0.5, zorder=1)

    # 绘制点：使用更鲜艳的渐变色映射
    # 'viridis' 或 'plasma' 在深色/白色背景下都很清晰
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=depths, cmap='plasma', s=40, 
                        alpha=0.8, edgecolors='none', zorder=2)

    # 标注根节点：略微放大字体
    for i, node in enumerate(all_nodes_info):
        if node['depth'] == 0:
            ax.text(coords_2d[i, 0], coords_2d[i, 1], node['name'], 
                    fontsize=11, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # 添加颜色条，确保颜色能被看到
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Hierarchy Depth', fontsize=12)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title("Balanced Poincaré Disk: AudioSet Hierarchy", fontsize=16)
    
    save_path = os.path.join(output_dir, "balanced_audio_hierarchy.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Balanced visualization saved to: {save_path}")
    plt.show()

def visualize_by_cluster(embeddings, all_nodes_info, output_dir="./Figures"):
    print("Generating Cluster-based Visualization with custom colors...")
    
    # 1. 坐标计算 (保持逻辑不变以确保两张图点位一致)
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.3, 
        metric=poincare_distance, 
        random_state=42,
        output_metric='euclidean' 
    )
    coords_2d = reducer.fit_transform(embeddings)
    coords_2d -= np.mean(coords_2d, axis=0)
    norms = np.linalg.norm(coords_2d, axis=1)
    r = np.clip(norms / (np.percentile(norms, 99) + 1e-7), 0, 1)
    new_r = np.power(r, 0.6)
    theta = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    coords_2d[:, 0] = new_r * np.cos(theta)
    coords_2d[:, 1] = new_r * np.sin(theta)

    # 2. 准备自定义颜色映射 (蓝, 绿, 红, 橙, 紫, 棕, 灰)
    custom_colors = [
        '#1f77b4', # 蓝色 (Blue)
        '#2ca02c', # 绿色 (Green)
        '#d62728', # 红色 (Red)
        '#ff7f0e', # 橙色 (Orange)
        '#9467bd', # 紫色 (Purple)
        '#8c564b', # 棕色 (Brown)
        '#7f7f7f', # 灰色 (Gray)
        '#e377c2', # 粉色 (作为备选)
        '#bcbd22', # 黄绿色
        '#17becf'  # 青色
    ]
    
    root_names = [n['root_name'] for n in all_nodes_info]
    unique_roots = list(dict.fromkeys(root_names)) 
    
    # 将 root 映射到自定义颜色
    root_to_color = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(unique_roots)}
    colors = [root_to_color[n['root_name']] for n in all_nodes_info]

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.add_artist(plt.Circle((0, 0), 1.0, color='#E0E0E0', fill=False, lw=2, zorder=0))

    # 绘制连线
    name_to_idx = {node['name']: i for i, node in enumerate(all_nodes_info)}
    for i, node in enumerate(all_nodes_info):
        if node['parent'] in name_to_idx:
            p_idx = name_to_idx[node['parent']]
            ax.plot([coords_2d[i, 0], coords_2d[p_idx, 0]], 
                    [coords_2d[i, 1], coords_2d[p_idx, 1]], 
                    color='gray', alpha=0.08, lw=0.5, zorder=1)

    # 绘制散点
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=colors, s=50, alpha=0.9, edgecolors='white', linewidths=0.3, zorder=2)

    # 标注根节点名称 (加上对应颜色的边框或文字颜色)
    for i, node in enumerate(all_nodes_info):
        if node['depth'] == 0:
            root_col = root_to_color[node['name']]
            ax.text(coords_2d[i, 0], coords_2d[i, 1], node['name'], 
                    fontsize=12, fontweight='bold', ha='center',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=root_col, boxstyle='round,pad=0.3', lw=2))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title("Poincaré Disk: Clustered by AudioSet Tree Roots (Custom Colors)", fontsize=16)
    
    save_path = os.path.join(output_dir, "cluster_audio_hierarchy_custom.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Custom cluster visualization saved to: {save_path}")
    plt.show()

## ==========================================
## 3. Main Execution
## ==========================================

if __name__ == "__main__":
    TREE_PATH = r"E:\Desktop\DDA4220\DDA4220_GroupProject\Datasets\AudioSet\tree.json"
    PROJ_CHECKPOINT = r"E:\Desktop\DDA4220\DDA4220_GroupProject\checkpoints_hyperbolic_c0_006_t1\best_projection_epoch_10.pth"
    CLAP_PATH = r"C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    model = ClapModel.from_pretrained(CLAP_PATH).to(device)
    processor = ClapProcessor.from_pretrained(CLAP_PATH)
    ckpt = torch.load(PROJ_CHECKPOINT, map_location=device)
    projection = HyperbolicProjection(input_dim=512, output_dim=512, c=ckpt.get('c', 1.0)).to(device)
    projection.load_state_dict(ckpt['projection_state_dict'])
    model.eval(); projection.eval()

    # Load Full JSON
    with open(TREE_PATH, 'r', encoding='utf-8') as f:
        full_tree = json.load(f)

    all_nodes_info = []
    all_embeddings = []

    # 1. 收集所有子树的所有节点
    print("Extracting embeddings for all trees...")
    for root_key, root_content in full_tree.items():
        sub_nodes = flatten_audio_tree(root_content, root_key, root_name=root_key)
        if len(sub_nodes) < 5:
                    print(f"Skipping '{root_key}': too few nodes ({len(sub_nodes)})")
                    continue

        all_nodes_info.extend(sub_nodes)
        
        # 批量提取该子树的文本 Embedding
        sub_names = [n['name'] for n in sub_nodes]
        with torch.no_grad():
            for name in tqdm(sub_names, desc=f"Processing {root_key}"):
                inputs = processor(text=[f"This is a sound of {name}."], return_tensors="pt", padding=True).to(device)
                embed = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
                hyp_embed = projection(embed).cpu().numpy()
                all_embeddings.append(hyp_embed)

    final_embeddings = np.vstack(all_embeddings)

    # 2. 统一可视化并保存
    visualize_combined_trees(np.vstack(all_embeddings), all_nodes_info)

    visualize_by_cluster(final_embeddings, all_nodes_info)