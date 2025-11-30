"""
CLAP音频嵌入t-SNE可视化脚本
使用CLAP提取音频嵌入，然后通过t-SNE降维到2维，按human_label进行可视化
"""
import argparse
import os
import glob
import json
import io
import pandas as pd
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter

from modelscope import ClapModel, ClapProcessor

# 可选的双曲投影支持
try:
    from hyperbolic_projection import HyperbolicProjection
    HYPERBOLIC_AVAILABLE = True
except ImportError:
    HYPERBOLIC_AVAILABLE = False


def process_audio_from_parquet(row):
    """Extract and process audio data from parquet row"""
    sample = {}
    
    # Read bytes from audio column
    if 'audio' in row and pd.notna(row['audio']):
        audio_val = row['audio']
        try:
            # Handle different audio data formats
            audio_bytes = None
            
            if isinstance(audio_val, dict):
                if 'bytes' in audio_val:
                    audio_bytes = audio_val['bytes']
                elif 'array' in audio_val:
                    audio = np.array(audio_val['array'], dtype=np.float32)
                    if audio.ndim > 1:
                        if audio.shape[0] > audio.shape[1]:
                            audio = np.mean(audio, axis=1)
                        else:
                            audio = np.mean(audio, axis=0)
                    audio = audio.flatten()
                    if len(audio) < 100:
                        sample['audio_error'] = f"Audio too short: {len(audio)}"
                        sample['has_audio'] = False
                        return sample
                    sample['audio'] = audio
                    sample['has_audio'] = True
                    return sample
            elif isinstance(audio_val, bytes):
                audio_bytes = audio_val
            elif isinstance(audio_val, (list, np.ndarray)):
                audio = np.array(audio_val, dtype=np.float32)
                if audio.ndim > 1:
                    if audio.shape[0] > audio.shape[1]:
                        audio = np.mean(audio, axis=1)
                    else:
                        audio = np.mean(audio, axis=0)
                audio = audio.flatten()
                if len(audio) < 100:
                    sample['audio_error'] = f"Audio too short: {len(audio)}"
                    sample['has_audio'] = False
                    return sample
                sample['audio'] = audio
                sample['has_audio'] = True
                return sample
            
            if audio_bytes is not None:
                audio_file_obj = io.BytesIO(audio_bytes)
                try:
                    audio, sr = sf.read(audio_file_obj)
                except Exception as sf_error:
                    try:
                        audio_file_obj.seek(0)
                        audio, sr = librosa.load(audio_file_obj, sr=None)
                    except Exception as librosa_error:
                        sample['audio_error'] = f"soundfile: {sf_error}, librosa: {librosa_error}"
                        sample['has_audio'] = False
                        return sample
                
                if len(audio) < 100:
                    sample['audio_error'] = f"Audio too short: {len(audio)}, sample rate: {sr}"
                    sample['has_audio'] = False
                    return sample
                
                if audio.ndim > 1:
                    if audio.shape[0] > audio.shape[1]:
                        audio = np.mean(audio, axis=1)
                    else:
                        audio = np.mean(audio, axis=0)
                audio = audio.flatten().astype(np.float32)
                
                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val
                elif max_val == 0:
                    sample['audio_error'] = "Audio is all zeros"
                    sample['has_audio'] = False
                    return sample
                
                if sr != 48000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                
                if len(audio) < 100:
                    sample['audio_error'] = f"Audio too short after resampling: {len(audio)}"
                    sample['has_audio'] = False
                    return sample
                
                if np.abs(audio).max() == 0:
                    sample['audio_error'] = "Audio is all zeros"
                    sample['has_audio'] = False
                    return sample
                
                sample['audio'] = audio
                sample['has_audio'] = True
                sample['sample_rate'] = 48000
                sample['audio_length'] = len(audio)
            else:
                sample['has_audio'] = False
                sample['audio_error'] = "Unable to recognize audio data format"
        except Exception as e:
            sample['audio_error'] = str(e)
            sample['has_audio'] = False
    else:
        sample['has_audio'] = False
    
    sample['video_id'] = row.get('video_id', '')
    
    # Get human_labels
    if 'human_labels' in row:
        human_labels = row['human_labels']
        if isinstance(human_labels, (list, np.ndarray)):
            if len(human_labels) > 0:
                valid_labels = [label for label in human_labels if pd.notna(label)]
                sample['human_labels'] = [str(label) for label in valid_labels] if valid_labels else []
            else:
                sample['human_labels'] = []
        elif human_labels is not None and pd.notna(human_labels):
            sample['human_labels'] = [str(human_labels)]
        else:
            sample['human_labels'] = []
    else:
        sample['human_labels'] = []
    
    return sample


def extract_audio_embeddings(model, processor, audio_samples, device, batch_size=16, projection=None):
    """Extract audio embeddings (optionally project to hyperbolic space)"""
    audio_embeds = []
    valid_indices = []
    
    use_hyperbolic = projection is not None
    if use_hyperbolic:
        print(f"  提取音频嵌入并投影到双曲空间 ({len(audio_samples)} 个样本)...")
    else:
        print(f"  提取音频嵌入 ({len(audio_samples)} 个样本)...")
    
    for batch_start in tqdm(range(0, len(audio_samples), batch_size), desc="    处理音频批次", unit="batch", leave=False):
        batch_end = min(batch_start + batch_size, len(audio_samples))
        batch_samples = audio_samples[batch_start:batch_end]
        
        batch_audios = []
        batch_valid_indices = []
        
        for i, sample in enumerate(batch_samples):
            idx = batch_start + i
            if not sample.get('has_audio', False):
                continue
            
            try:
                audio = sample['audio']
                
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)
                
                if audio.ndim > 1:
                    audio = audio.flatten()
                
                if len(audio) < 100:
                    continue
                
                batch_audios.append(audio)
                batch_valid_indices.append(idx)
            except Exception as e:
                continue
        
        if not batch_audios:
            continue
        
        try:
            inputs = processor(audios=batch_audios, sampling_rate=48000, return_tensors="pt").to(device)
            
            with torch.no_grad():
                audio_embed_batch = model.get_audio_features(**inputs)
                
                if audio_embed_batch.dim() == 1:
                    audio_embed_batch = audio_embed_batch.unsqueeze(0)
                
                audio_embed_batch = torch.nn.functional.normalize(audio_embed_batch, dim=-1)
                
                if projection is not None:
                    audio_embed_batch = projection(audio_embed_batch)
                
                for j, embed in enumerate(audio_embed_batch):
                    if j >= len(batch_valid_indices):
                        break
                    
                    embed_np = embed.cpu().numpy()
                    
                    if embed_np.ndim > 1:
                        embed_np = embed_np.squeeze()
                    
                    audio_embeds.append(embed_np)
                    valid_indices.append(batch_valid_indices[j])
        except Exception as e:
            print(f"\n    ⚠ 批次 {batch_start//batch_size} 处理失败: {e}")
            continue
    
    if audio_embeds:
        audio_embeds = np.vstack(audio_embeds)
        print(f"  ✓ 成功提取 {len(audio_embeds)} 个音频嵌入，形状: {audio_embeds.shape}")
        return audio_embeds, valid_indices
    else:
        print("  ✗ 未能提取任何音频嵌入")
        return None, []


def visualize_tsne(audio_embeds, human_labels_list, valid_indices, samples, output_path, 
                   perplexity=30, n_iter=1000, random_state=42, max_samples_per_label=None,
                   learning_rate=200.0, early_exaggeration=12.0, min_grad_norm=1e-7,
                   metric='euclidean', init='random', n_jobs=None,
                   figsize=(16, 12), dpi=300, point_size=30, alpha=0.6, colormap='tab20'):
    """
    使用t-SNE降维并可视化
    
    Args:
        audio_embeds: 音频嵌入矩阵 (n_samples, embed_dim)
        human_labels_list: 每个样本的human_labels列表
        valid_indices: 有效样本索引
        samples: 原始样本列表
        output_path: 输出图片路径
        perplexity: t-SNE的perplexity参数（通常5-50，建议值：5-50，样本数少时用较小值）
        n_iter: t-SNE最大迭代次数（传递给TSNE的max_iter参数）
        random_state: 随机种子
        max_samples_per_label: 每个标签最多使用的样本数（用于平衡数据）
        learning_rate: 学习率（默认200，通常10-1000，样本数多时用较大值）
        early_exaggeration: 早期放大因子（默认12，影响早期迭代的分离度，通常4-20）
        min_grad_norm: 最小梯度范数（默认1e-7，梯度小于此值时停止）
        metric: 距离度量方式（'euclidean', 'manhattan', 'chebyshev', 'minkowski'等）
        init: 初始化方式（'random'或'pca'，pca通常更稳定但更慢）
        n_jobs: 并行数（None表示使用所有CPU核心）
        figsize: 图形大小（宽, 高）英寸
        dpi: 图片分辨率（每英寸点数，默认300）
        point_size: 散点图中点的大小（默认30）
        alpha: 点的透明度（0-1，默认0.6）
        colormap: 颜色映射名称（'tab20', 'tab10', 'Set3', 'viridis'等）
    """
    print(f"\n  准备t-SNE可视化...")
    print(f"    嵌入维度: {audio_embeds.shape}")
    print(f"    样本数量: {len(audio_embeds)}")
    
    # 准备标签数据
    # 对于每个样本，使用第一个human_label作为主要标签
    labels = []
    label_to_samples = {}
    
    for embed_idx, sample_idx in enumerate(valid_indices):
        if sample_idx >= len(samples):
            continue
        
        sample = samples[sample_idx]
        human_labels = sample.get('human_labels', [])
        
        if human_labels:
            # 使用第一个human_label
            main_label = str(human_labels[0]).strip()
            labels.append(main_label)
            
            if main_label not in label_to_samples:
                label_to_samples[main_label] = []
            label_to_samples[main_label].append(embed_idx)
        else:
            # 如果没有human_label，使用"未知"
            labels.append("未知")
            if "未知" not in label_to_samples:
                label_to_samples["未知"] = []
            label_to_samples["未知"].append(embed_idx)
    
    # 统计标签分布
    label_counts = Counter(labels)
    print(f"\n  标签统计:")
    for label, count in label_counts.most_common(20):
        print(f"    {label}: {count} 个样本")
    
    # 如果指定了max_samples_per_label，进行采样平衡
    if max_samples_per_label is not None and max_samples_per_label > 0:
        print(f"\n  平衡数据（每个标签最多 {max_samples_per_label} 个样本）...")
        selected_indices = []
        selected_labels = []
        
        for label, indices in label_to_samples.items():
            if len(indices) > max_samples_per_label:
                # 随机采样
                np.random.seed(random_state)
                sampled_indices = np.random.choice(indices, size=max_samples_per_label, replace=False)
                selected_indices.extend(sampled_indices.tolist())
                selected_labels.extend([label] * len(sampled_indices))
            else:
                selected_indices.extend(indices)
                selected_labels.extend([label] * len(indices))
        
        # 重新排序以保持索引顺序
        sorted_pairs = sorted(zip(selected_indices, selected_labels))
        selected_indices, selected_labels = zip(*sorted_pairs)
        selected_indices = list(selected_indices)
        selected_labels = list(selected_labels)
        
        audio_embeds_filtered = audio_embeds[selected_indices]
        labels_filtered = selected_labels
        
        print(f"    原始样本数: {len(audio_embeds)}")
        print(f"    平衡后样本数: {len(audio_embeds_filtered)}")
    else:
        audio_embeds_filtered = audio_embeds
        labels_filtered = labels
    
    # 运行t-SNE
    print(f"\n  运行t-SNE降维...")
    print(f"    参数: perplexity={perplexity}, max_iter={n_iter}, learning_rate={learning_rate}")
    print(f"          early_exaggeration={early_exaggeration}, metric={metric}, init={init}")
    
    tsne_kwargs = {
        'n_components': 2,
        'perplexity': perplexity,
        'max_iter': n_iter,
        'random_state': random_state,
        'verbose': 1,
        'learning_rate': learning_rate,
        'early_exaggeration': early_exaggeration,
        'min_grad_norm': min_grad_norm,
        'metric': metric,
        'init': init
    }
    
    if n_jobs is not None:
        tsne_kwargs['n_jobs'] = n_jobs
    
    tsne = TSNE(**tsne_kwargs)
    embeddings_2d = tsne.fit_transform(audio_embeds_filtered)
    
    print(f"  ✓ t-SNE完成，2D嵌入形状: {embeddings_2d.shape}")
    
    # 准备可视化
    unique_labels = sorted(set(labels_filtered))
    n_labels = len(unique_labels)
    
    # 生成颜色映射
    try:
        cmap = plt.cm.get_cmap(colormap)
    except ValueError:
        print(f"  ⚠ 警告: 颜色映射 '{colormap}' 不存在，使用默认 'tab20'")
        cmap = plt.cm.get_cmap('tab20')
    
    colors = [cmap(i / max(n_labels, 1)) for i in range(n_labels)]
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制每个标签的点
    for label in unique_labels:
        mask = np.array(labels_filtered) == label
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        plt.scatter(x, y, c=[label_to_color[label]], label=label, alpha=alpha, s=point_size)
    
    plt.xlabel('t-SNE维度 1', fontsize=12)
    plt.ylabel('t-SNE维度 2', fontsize=12)
    plt.title(f'CLAP音频嵌入t-SNE可视化 (样本数: {len(embeddings_2d)})', fontsize=14, fontweight='bold')
    
    # 图例
    if n_labels <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # 如果标签太多，只显示前20个最常见的
        top_labels = [label for label, _ in Counter(labels_filtered).most_common(20)]
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        filtered_handles = [h for h, l in zip(handles, labels_legend) if l in top_labels]
        filtered_labels_legend = [l for l in labels_legend if l in top_labels]
        plt.legend(filtered_handles, filtered_labels_legend, bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ 可视化结果已保存到: {output_path} (DPI: {dpi})")
    
    plt.close()
    
    # 返回统计信息
    stats = {
        'total_samples': len(audio_embeds),
        'visualized_samples': len(embeddings_2d),
        'num_labels': n_labels,
        'label_distribution': dict(Counter(labels_filtered))
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='CLAP音频嵌入t-SNE可视化')
    parser.add_argument('--parquet-dir', type=str, required=True, help='Parquet文件目录')
    parser.add_argument('--num-samples', type=int, default=1000, help='使用的样本数量')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备 (cuda:0 或 cpu)')
    parser.add_argument('--checkpoint', type=str, 
                        default=r'C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general',
                        help='ModelScope模型路径')
    parser.add_argument('--output-image', type=str, default=None, help='输出图片路径')
    parser.add_argument('--use-hyperbolic', action='store_true', help='使用双曲投影')
    parser.add_argument('--projection-checkpoint', type=str, default=None, 
                        help='双曲投影检查点路径（使用--use-hyperbolic时需要）')
    parser.add_argument('--c', type=float, default=1.0, 
                        help='双曲空间曲率参数（仅在使用--use-hyperbolic时使用）')
    parser.add_argument('--perplexity', type=int, default=30, 
                        help='t-SNE的perplexity参数（通常5-50，样本数少时用较小值）')
    parser.add_argument('--n-iter', type=int, default=1000, help='t-SNE最大迭代次数')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    parser.add_argument('--max-samples-per-label', type=int, default=None, 
                        help='每个标签最多使用的样本数（用于平衡数据）')
    
    # t-SNE高级参数
    parser.add_argument('--learning-rate', type=float, default=200.0,
                        help='t-SNE学习率（通常10-1000，样本数多时用较大值，默认200）')
    parser.add_argument('--early-exaggeration', type=float, default=12.0,
                        help='t-SNE早期放大因子（通常4-20，影响早期迭代的分离度，默认12）')
    parser.add_argument('--min-grad-norm', type=float, default=1e-7,
                        help='t-SNE最小梯度范数（梯度小于此值时停止，默认1e-7）')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                        help='距离度量方式（默认euclidean）')
    parser.add_argument('--init', type=str, default='random',
                        choices=['random', 'pca'],
                        help='初始化方式（random或pca，pca通常更稳定但更慢，默认random）')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='并行数（None表示使用所有CPU核心，默认None）')
    
    # 可视化参数
    parser.add_argument('--figsize', type=str, default='16,12',
                        help='图形大小，格式：宽,高（英寸），默认16,12')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图片分辨率DPI（每英寸点数，默认300）')
    parser.add_argument('--point-size', type=float, default=30,
                        help='散点图中点的大小（默认30）')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='点的透明度（0-1，默认0.6）')
    parser.add_argument('--colormap', type=str, default='tab20',
                        help='颜色映射名称（tab20, tab10, Set3, viridis等，默认tab20）')
    
    args = parser.parse_args()
    
    # 验证双曲参数
    if args.use_hyperbolic:
        if not HYPERBOLIC_AVAILABLE:
            print("⚠ 警告: 指定了--use-hyperbolic但双曲投影模块不可用")
            print("  回退到标准CLAP")
            args.use_hyperbolic = False
        elif args.projection_checkpoint is None or not os.path.exists(args.projection_checkpoint):
            print("⚠ 警告: 指定了--use-hyperbolic但未找到投影检查点")
            print(f"  检查点路径: {args.projection_checkpoint}")
            print("  回退到标准CLAP")
            args.use_hyperbolic = False
    
    # 检查设备
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"⚠ 警告: 指定了 {args.device}，但CUDA不可用，切换到CPU")
            args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n[1/4] 加载ModelScope CLAP模型...")
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_path = args.checkpoint
        print(f"  从本地路径加载: {model_path}")
    else:
        if args.checkpoint:
            print(f"  ⚠ 警告: 指定的检查点路径不存在: {args.checkpoint}")
            print(f"  回退到模型ID: laion/larger_clap_general")
        model_path = "laion/larger_clap_general"
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "laion" / "larger_clap_general"
        if cache_dir.exists():
            print(f"  使用模型ID: {model_path}")
            print(f"  ✓ 检测到本地缓存，将直接使用: {cache_dir}")
        else:
            print(f"  使用模型ID: {model_path}")
            print(f"  ℹ ModelScope将自动检查本地缓存，如果不存在则下载")
    
    try:
        model = ClapModel.from_pretrained(model_path).to(device)
        processor = ClapProcessor.from_pretrained(model_path)
        if hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            processor.feature_extractor.sampling_rate = 48000
        model.eval()
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载双曲投影（如果需要）
    projection = None
    if args.use_hyperbolic:
        print("\n[1.5/4] 加载双曲投影层...")
        try:
            checkpoint = torch.load(args.projection_checkpoint, map_location=device)
            embed_dim = checkpoint.get('embed_dim', 512)
            c = checkpoint.get('c', args.c)
            
            projection = HyperbolicProjection(
                input_dim=embed_dim,
                output_dim=embed_dim,
                c=c,
                clip_r=0.9
            ).to(device)
            
            projection.load_state_dict(checkpoint['projection_state_dict'])
            projection.eval()
            
            print(f"  ✓ 双曲投影加载成功")
            print(f"    嵌入维度: {embed_dim}")
            print(f"    曲率参数: {c}")
            args.c = c
        except Exception as e:
            print(f"  ✗ 双曲投影加载失败: {e}")
            import traceback
            traceback.print_exc()
            print("  回退到标准CLAP")
            args.use_hyperbolic = False
            projection = None
    
    # 2. 读取parquet文件
    print(f"\n[2/4] 读取parquet文件...")
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"  ✗ 未找到parquet文件")
        return
    
    print(f"  ✓ 找到 {len(parquet_files)} 个parquet文件")
    
    # 3. 准备数据
    print(f"\n[3/4] 准备数据...")
    test_samples = []
    
    pbar = tqdm(total=args.num_samples, desc="  读取样本", unit="sample", leave=False)
    
    # 读取多个parquet文件，直到达到所需样本数
    samples_before_file = 0
    for parquet_file in parquet_files:
        if len(test_samples) >= args.num_samples:
            break
        
        samples_before_file = len(test_samples)
        print(f"  正在读取文件: {os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            if len(test_samples) >= args.num_samples:
                break
            
            sample = process_audio_from_parquet(row)
            if sample.get('has_audio', False):
                test_samples.append(sample)
                pbar.update(1)
        
        samples_from_file = len(test_samples) - samples_before_file
        print(f"  已从 {os.path.basename(parquet_file)} 读取 {samples_from_file} 个有效样本")
    
    pbar.close()
    
    samples_with_audio = sum(1 for s in test_samples if s.get('has_audio', False))
    print(f"  ✓ 准备了 {len(test_samples)} 个测试样本 ({samples_with_audio} 个有音频)")
    
    if not test_samples:
        print("  ✗ 未找到有效的测试样本")
        return
    
    # 4. 提取音频嵌入
    print(f"\n[4/4] 提取音频嵌入...")
    audio_samples = [s for s in test_samples if s.get('has_audio', False)]
    audio_embeds, valid_indices = extract_audio_embeddings(
        model, processor, audio_samples, device, projection=projection
    )
    
    if audio_embeds is None:
        print("  ✗ 无法提取音频嵌入，程序终止")
        return
    
    # 5. t-SNE可视化
    print(f"\n[5/4] 进行t-SNE可视化...")
    
    # 准备human_labels列表
    human_labels_list = []
    for sample_idx in valid_indices:
        if sample_idx < len(test_samples):
            sample = test_samples[sample_idx]
            human_labels_list.append(sample.get('human_labels', []))
    
    # 确定输出路径
    output_dir = "tSNE"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output_image:
        # 如果用户指定了输出路径，确保目录存在
        output_path = args.output_image
        output_dir_user = os.path.dirname(output_path)
        if output_dir_user:
            os.makedirs(output_dir_user, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = "hyperbolic" if args.use_hyperbolic else "standard"
        output_path = os.path.join(output_dir, f"clap_tsne_visualization_{model_type}_{timestamp}.png")
    
    # 解析figsize参数
    try:
        figsize_parts = args.figsize.split(',')
        figsize = (float(figsize_parts[0].strip()), float(figsize_parts[1].strip()))
    except:
        print(f"  ⚠ 警告: 无法解析figsize参数 '{args.figsize}'，使用默认值 (16, 12)")
        figsize = (16, 12)
    
    # 执行可视化
    stats = visualize_tsne(
        audio_embeds, human_labels_list, valid_indices, test_samples,
        output_path, 
        perplexity=args.perplexity, 
        n_iter=args.n_iter,
        random_state=args.random_state, 
        max_samples_per_label=args.max_samples_per_label,
        learning_rate=args.learning_rate,
        early_exaggeration=args.early_exaggeration,
        min_grad_norm=args.min_grad_norm,
        metric=args.metric,
        init=args.init,
        n_jobs=args.n_jobs,
        figsize=figsize,
        dpi=args.dpi,
        point_size=args.point_size,
        alpha=args.alpha,
        colormap=args.colormap
    )
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"可视化统计信息:")
    print(f"{'='*60}")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  可视化样本数: {stats['visualized_samples']}")
    print(f"  标签数量: {stats['num_labels']}")
    print(f"\n  标签分布（前10个）:")
    for label, count in sorted(stats['label_distribution'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {label}: {count}")
    print(f"{'='*60}\n")
    
    print(f"✓ 完成！可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

