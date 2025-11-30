"""
双曲CLAP ModelScope零样本分类测试脚本
使用双曲投影层和双曲相似度进行零样本分类
"""
import argparse
import os
import glob
import json
import io
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from datetime import datetime

from modelscope import ClapModel, ClapProcessor
from hyperbolic_projection import HyperbolicProjection, hyperbolic_similarity_matrix_batch


def load_audioset_classes(class_labels_path):
    """加载AudioSet类别标签"""
    if not os.path.exists(class_labels_path):
        print(f"⚠ 警告: 类别文件不存在: {class_labels_path}")
        return None
    
    with open(class_labels_path, 'r', encoding='utf-8') as f:
        class_index_dict = json.load(f)
    
    classnames = list(class_index_dict.keys())
    print(f"✓ 加载了 {len(classnames)} 个AudioSet类别")
    return classnames


def process_audio_from_parquet(row):
    """从parquet行中提取和处理音频数据"""
    sample = {}
    
    if 'audio' in row and pd.notna(row['audio']):
        audio_val = row['audio']
        try:
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
                        sample['audio_error'] = f"音频太短: {len(audio)}"
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
                    sample['audio_error'] = f"音频太短: {len(audio)}"
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
                    sample['audio_error'] = f"音频太短: {len(audio)}, 采样率: {sr}"
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
                    sample['audio_error'] = "音频全为零"
                    sample['has_audio'] = False
                    return sample
                
                if sr != 48000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                
                if len(audio) < 100 or np.abs(audio).max() == 0:
                    sample['audio_error'] = f"音频太短或全为零: {len(audio)}"
                    sample['has_audio'] = False
                    return sample
                
                sample['audio'] = audio
                sample['has_audio'] = True
                sample['sample_rate'] = 48000
            else:
                sample['has_audio'] = False
                sample['audio_error'] = "无法识别音频数据格式"
        except Exception as e:
            sample['audio_error'] = str(e)
            sample['has_audio'] = False
    else:
        sample['has_audio'] = False
    
    sample['video_id'] = row.get('video_id', '')
    
    if 'labels' in row:
        labels = row['labels']
        if isinstance(labels, (list, np.ndarray)):
            valid_labels = [label for label in labels if pd.notna(label)]
            sample['labels'] = [str(label) for label in valid_labels] if valid_labels else []
        elif labels is not None and pd.notna(labels):
            sample['labels'] = [str(labels)]
        else:
            sample['labels'] = []
    else:
        sample['labels'] = []
    
    if 'human_labels' in row:
        human_labels = row['human_labels']
        if isinstance(human_labels, (list, np.ndarray)):
            valid_labels = [label for label in human_labels if pd.notna(label)]
            sample['human_labels'] = [str(label) for label in valid_labels] if valid_labels else []
        elif human_labels is not None and pd.notna(human_labels):
            sample['human_labels'] = [str(human_labels)]
        else:
            sample['human_labels'] = []
    else:
        sample['human_labels'] = []
    
    return sample


def extract_audio_embeddings(model, processor, projection, audio_samples, device, batch_size=16, c=1.0):
    """提取音频嵌入并投影到双曲空间"""
    audio_embeds = []
    valid_indices = []
    
    print(f"  提取音频嵌入并投影到双曲空间 ({len(audio_samples)} 样本)...")
    
    model.eval()
    projection.eval()
    
    for batch_start in tqdm(range(0, len(audio_samples), batch_size), desc="    处理音频批次", unit="批次", leave=False):
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
            with torch.no_grad():
                # 提取CLAP音频嵌入
                inputs = processor(audios=batch_audios, sampling_rate=48000, return_tensors="pt").to(device)
                audio_embed_batch = model.get_audio_features(**inputs)
                
                if audio_embed_batch.dim() == 1:
                    audio_embed_batch = audio_embed_batch.unsqueeze(0)
                
                # 归一化
                audio_embed_batch = F.normalize(audio_embed_batch, dim=-1)
                
                # 投影到双曲空间
                audio_hyperbolic = projection(audio_embed_batch)
                
                # 转换为numpy
                for j, embed in enumerate(audio_hyperbolic):
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
        print(f"  ✓ 成功提取 {len(audio_embeds)} 个双曲音频嵌入，形状: {audio_embeds.shape}")
        return audio_embeds, valid_indices
    else:
        print("  ✗ 未能提取任何音频嵌入")
        return None, []


def extract_text_embeddings(model, processor, projection, texts, device, batch_size=16, c=1.0):
    """提取文本嵌入并投影到双曲空间"""
    text_embeds = []
    
    print(f"  提取文本嵌入并投影到双曲空间 ({len(texts)} 文本)...")
    
    model.eval()
    projection.eval()
    
    pbar = tqdm(total=len(texts), desc="    处理文本", unit="文本", leave=False)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            with torch.no_grad():
                # 提取CLAP文本嵌入
                inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embed = model.get_text_features(**inputs)
                
                # 归一化
                text_embed = F.normalize(text_embed, dim=-1)
                
                # 投影到双曲空间
                text_hyperbolic = projection(text_embed)
                
                text_embeds.append(text_hyperbolic.cpu().numpy())
        except Exception as e:
            print(f"\n    ⚠ 批次 {i//batch_size} 处理失败: {e}")
            continue
        
        pbar.update(len(batch_texts))
    
    pbar.close()
    
    if text_embeds:
        text_embeds = np.vstack(text_embeds)
        print(f"  ✓ 成功提取 {len(text_embeds)} 个双曲文本嵌入")
        return text_embeds
    else:
        print("  ✗ 未能提取任何文本嵌入")
        return None


def calculate_hyperbolic_similarity(audio_embeds, text_embeds, c=1.0, temperature=1.0):
    """计算双曲相似度矩阵"""
    audio_tensor = torch.from_numpy(audio_embeds).float()
    text_tensor = torch.from_numpy(text_embeds).float()
    
    # 计算双曲相似度矩阵
    similarity = hyperbolic_similarity_matrix_batch(
        audio_tensor, text_tensor, c=c, temperature=temperature
    )
    
    return similarity.numpy()


def calculate_metrics(predictions, classnames, similarity_matrix=None, valid_indices=None, top_k_values=[1, 3, 5, 10]):
    """计算评估指标：top-k准确率、F1分数和mAP"""
    classname_to_idx = {name: idx for idx, name in enumerate(classnames)}
    
    valid_predictions = [p for p in predictions if p.get('ground_truth_human_labels', [])]
    
    if not valid_predictions:
        return None
    
    num_classes = len(classnames)
    num_samples = len(valid_predictions)
    
    pred_idx_to_valid_idx = {}
    valid_idx = 0
    for i, pred in enumerate(predictions):
        if pred.get('ground_truth_human_labels', []):
            pred_idx_to_valid_idx[i] = valid_idx
            valid_idx += 1
    
    y_true = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_pred_topk = {}
    
    for k in top_k_values:
        y_pred_topk[k] = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    matched_count = 0
    unmatched_labels = []
    
    for i, pred in enumerate(valid_predictions):
        gt_labels = pred.get('ground_truth_human_labels', [])
        if not gt_labels:
            gt_labels = pred.get('ground_truth_labels', [])
        
        sample_matched = False
        for label in gt_labels:
            label_str = str(label).strip().lower()
            found_match = False
            for classname, idx in classname_to_idx.items():
                if classname.strip().lower() == label_str:
                    y_true[i, idx] = 1.0
                    found_match = True
                    sample_matched = True
                    break
            
            if not found_match:
                unmatched_labels.append(label_str)
        
        if sample_matched:
            matched_count += 1
    
    if unmatched_labels:
        unique_unmatched = list(set(unmatched_labels))[:10]
        print(f"  调试: {len(unmatched_labels)} 个真实标签无法匹配到类别名")
        print(f"  调试: 未匹配标签示例: {unique_unmatched}")
    
    print(f"  调试: {matched_count}/{num_samples} 个样本至少有一个匹配的真实标签")
    
    for i, pred in enumerate(valid_predictions):
        top_k_preds = pred.get('top_k_predictions', [])
        for k in top_k_values:
            for j in range(min(k, len(top_k_preds))):
                pred_classname = top_k_preds[j]['classname']
                if pred_classname in classname_to_idx:
                    y_pred_topk[k][i, classname_to_idx[pred_classname]] = 1.0
    
    metrics = {}
    
    for k in top_k_values:
        correct = 0
        for i in range(num_samples):
            if np.any(np.logical_and(y_true[i], y_pred_topk[k][i])):
                correct += 1
        metrics[f'acc@{k}'] = correct / num_samples if num_samples > 0 else 0.0
    
    k_for_f1 = min(5, max(top_k_values))
    y_pred_f1 = y_pred_topk[k_for_f1]
    
    tp_micro = np.sum(np.logical_and(y_true, y_pred_f1))
    fp_micro = np.sum(np.logical_and(np.logical_not(y_true), y_pred_f1))
    fn_micro = np.sum(np.logical_and(y_true, np.logical_not(y_pred_f1)))
    
    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
    
    metrics['f1_micro'] = float(f1_micro)
    metrics['precision_micro'] = float(precision_micro)
    metrics['recall_micro'] = float(recall_micro)
    
    precisions_per_class = []
    recalls_per_class = []
    for c in range(num_classes):
        tp = np.sum(np.logical_and(y_true[:, c], y_pred_f1[:, c]))
        fp = np.sum(np.logical_and(np.logical_not(y_true[:, c]), y_pred_f1[:, c]))
        fn = np.sum(np.logical_and(y_true[:, c], np.logical_not(y_pred_f1[:, c])))
        
        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions_per_class.append(precision_c)
        recalls_per_class.append(recall_c)
    
    precision_macro = np.mean(precisions_per_class) if precisions_per_class else 0.0
    recall_macro = np.mean(recalls_per_class) if recalls_per_class else 0.0
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (precision_macro + recall_macro) > 0 else 0.0
    
    metrics['f1_macro'] = float(f1_macro)
    metrics['precision_macro'] = float(precision_macro)
    metrics['recall_macro'] = float(recall_macro)
    
    aps = []
    if similarity_matrix is not None and valid_indices is not None:
        sample_id_to_embed_idx = {}
        for embed_idx, sample_idx in enumerate(valid_indices):
            sample_id_to_embed_idx[sample_idx] = embed_idx
        
        for i, pred in enumerate(valid_predictions):
            sample_id = pred.get('sample_id')
            if sample_id is None or sample_id not in sample_id_to_embed_idx:
                continue
            
            embed_idx = sample_id_to_embed_idx[sample_id]
            if embed_idx >= similarity_matrix.shape[0]:
                continue
            
            sim_scores = similarity_matrix[embed_idx, :]
            
            gt_indices = []
            gt_labels = pred.get('ground_truth_human_labels', [])
            if not gt_labels:
                gt_labels = pred.get('ground_truth_labels', [])
            
            for label in gt_labels:
                label_str = str(label).strip().lower()
                for classname, idx in classname_to_idx.items():
                    if classname.strip().lower() == label_str:
                        gt_indices.append(idx)
                        break
            
            if not gt_indices:
                continue
            
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            num_relevant = len(gt_indices)
            relevant_found = 0
            precision_at_k = []
            
            for rank, class_idx in enumerate(sorted_indices):
                if class_idx in gt_indices:
                    relevant_found += 1
                    precision_at_k.append(relevant_found / (rank + 1))
            
            ap = np.mean(precision_at_k) if precision_at_k else 0.0
            aps.append(ap)
    else:
        for i in range(num_samples):
            if np.sum(y_true[i]) == 0:
                continue
            
            sample_pred = valid_predictions[i]
            top_k_preds = sample_pred.get('top_k_predictions', [])
            
            relevance = np.zeros(num_classes)
            for pred_item in top_k_preds:
                pred_classname = pred_item['classname']
                if pred_classname in classname_to_idx:
                    relevance[classname_to_idx[pred_classname]] = 1.0
            
            num_relevant = int(np.sum(y_true[i]))
            relevant_in_topk = np.sum(np.logical_and(y_true[i], relevance))
            if relevant_in_topk == 0:
                ap = 0.0
            else:
                ap = relevant_in_topk / min(len(top_k_preds), num_relevant)
            aps.append(ap)
    
    metrics['map'] = float(np.mean(aps)) if aps else 0.0
    metrics['valid_samples'] = num_samples
    metrics['total_samples'] = len(predictions)
    
    return metrics


def predict_classification(audio_embeds, class_text_embeds, classnames, test_samples, valid_indices, top_k=5, c=1.0, temperature=1.0):
    """使用双曲相似度进行零样本分类预测"""
    print(f"\n  计算双曲相似度矩阵...")
    similarity_matrix = calculate_hyperbolic_similarity(audio_embeds, class_text_embeds, c=c, temperature=temperature)
    print(f"  ✓ 相似度矩阵形状: {similarity_matrix.shape}")
    
    sim_min, sim_max = float(similarity_matrix.min()), float(similarity_matrix.max())
    sim_mean = float(similarity_matrix.mean())
    print(f"  相似度范围: [{sim_min:.4f}, {sim_max:.4f}], 均值: {sim_mean:.4f}")
    
    print(f"\n  生成分类预测...")
    predictions = []
    
    for embed_idx, sample_idx in enumerate(valid_indices):
        if sample_idx >= len(test_samples):
            continue
        
        sample = test_samples[sample_idx]
        
        audio_class_sims = similarity_matrix[embed_idx, :]
        ranking = np.argsort(audio_class_sims)[::-1]
        
        top_k_indices = ranking[:top_k]
        top_k_predictions = []
        
        for rank, class_idx in enumerate(top_k_indices):
            top_k_predictions.append({
                "rank": rank + 1,
                "classname": classnames[class_idx],
                "similarity": float(audio_class_sims[class_idx])
            })
        
        ground_truth_labels = sample.get('labels', [])
        ground_truth_human_labels = sample.get('human_labels', [])
        
        predicted_top1 = classnames[ranking[0]]
        
        predictions.append({
            "sample_id": sample_idx,
            "video_id": sample.get('video_id', ''),
            "predicted_top1": predicted_top1,
            "top_k_predictions": top_k_predictions,
            "ground_truth_labels": ground_truth_labels,
            "ground_truth_human_labels": ground_truth_human_labels
        })
    
    print(f"  ✓ 生成了 {len(predictions)} 个预测结果")
    
    return predictions, similarity_matrix


def main():
    parser = argparse.ArgumentParser(description='双曲CLAP ModelScope零样本分类测试')
    parser.add_argument('--parquet-dir', type=str, required=True, help='Parquet文件目录（eval）')
    parser.add_argument('--projection-checkpoint', type=str, required=True, help='双曲投影层检查点路径')
    parser.add_argument('--num-samples', type=int, default=100, help='测试样本数量')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备（cuda:0或cpu）')
    parser.add_argument('--checkpoint', type=str, 
                        default=r'C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general',
                        help='ModelScope模型路径（默认使用本地缓存路径）')
    parser.add_argument('--output-json', type=str, default=None, help='输出JSON文件路径')
    parser.add_argument('--class-labels', type=str,
                        default='CLAP/class_labels/audioset_class_labels_indices.json',
                        help='AudioSet类别标签文件路径')
    parser.add_argument('--top-k', type=int, default=5, help='输出top-k预测结果，默认5')
    parser.add_argument('--c', type=float, default=1.0, help='双曲空间曲率参数')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    
    args = parser.parse_args()
    
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"⚠ 警告: 指定了 {args.device}，但CUDA不可用，切换到CPU")
            args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 1. 加载CLAP模型
    print("\n[1/5] 加载ModelScope CLAP模型...")
    
    # 确定模型路径：优先使用提供的checkpoint，如果不存在则回退到模型ID
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_path = args.checkpoint
        print(f"  从本地路径加载: {model_path}")
    else:
        # 如果默认路径不存在，回退到模型ID（ModelScope会自动检查缓存）
        if args.checkpoint:
            print(f"  ⚠ 警告: 指定的checkpoint路径不存在: {args.checkpoint}")
            print(f"  回退到模型ID: laion/larger_clap_general")
        model_path = "laion/larger_clap_general"
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "laion" / "larger_clap_general"
        if cache_dir.exists():
            print(f"  使用模型ID: {model_path}")
            print(f"  ✓ 检测到本地缓存，将直接使用: {cache_dir}")
        else:
            print(f"  使用模型ID: {model_path}")
            print(f"  ℹ ModelScope将自动检查本地缓存，如果不存在才会下载")
    
    try:
        model = ClapModel.from_pretrained(model_path).to(device)
        processor = ClapProcessor.from_pretrained(model_path)
        model.eval()
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 加载双曲投影层
    print("\n[2/5] 加载双曲投影层...")
    if not os.path.exists(args.projection_checkpoint):
        print(f"  ✗ 投影层检查点不存在: {args.projection_checkpoint}")
        return
    
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
        
        print(f"  ✓ 投影层加载成功")
        print(f"    嵌入维度: {embed_dim}")
        print(f"    曲率参数: {c}")
    except Exception as e:
        print(f"  ✗ 投影层加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 读取parquet文件
    print(f"\n[3/5] 读取parquet文件...")
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"  ✗ 未找到parquet文件")
        return
    
    print(f"  ✓ 找到 {len(parquet_files)} 个parquet文件")
    
    # 4. 准备测试数据
    print(f"\n[4/5] 准备测试数据...")
    test_samples = []
    
    pbar = tqdm(total=min(args.num_samples, 1000), desc="  读取样本", unit="样本", leave=False)
    
    for parquet_file in parquet_files[:1]:  # 只读取第一个文件
        df = pd.read_parquet(parquet_file)
        for idx, row in df.iterrows():
            if len(test_samples) >= args.num_samples:
                break
            
            sample = process_audio_from_parquet(row)
            if sample.get('has_audio', False):
                test_samples.append(sample)
                pbar.update(1)
        
        if len(test_samples) >= args.num_samples:
            break
    
    pbar.close()
    
    samples_with_audio = sum(1 for s in test_samples if s.get('has_audio', False))
    print(f"  ✓ 准备了 {len(test_samples)} 个测试样本（{samples_with_audio} 个有音频）")
    
    if not test_samples:
        print("  ✗ 没有有效的测试样本")
        return
    
    # 5. 运行测试
    print(f"\n[5/5] 运行零样本分类测试...")
    
    # 5.1 提取音频嵌入
    audio_samples = [s for s in test_samples if s.get('has_audio', False)]
    audio_embeds, valid_indices = extract_audio_embeddings(
        model, processor, projection, audio_samples, device, c=c
    )
    
    if audio_embeds is None:
        print("  ✗ 无法提取音频嵌入，测试终止")
        return
    
    # 5.2 加载类别并提取文本嵌入
    classnames = load_audioset_classes(args.class_labels)
    if classnames is None:
        print("  ✗ 无法加载类别标签，测试终止")
        return
    
    class_texts = [f"This is a sound of {classname}." for classname in classnames]
    class_text_embeds = extract_text_embeddings(
        model, processor, projection, class_texts, device, c=c
    )
    
    if class_text_embeds is None:
        print("  ✗ 无法提取文本嵌入，测试终止")
        return
    
    # 5.3 生成分类预测
    predictions, similarity_matrix = predict_classification(
        audio_embeds, class_text_embeds, classnames,
        test_samples, valid_indices, top_k=args.top_k, c=c, temperature=args.temperature
    )
    
    if not predictions:
        print("  ✗ 未能生成预测结果")
        return
    
    # 6. 计算评估指标
    print(f"\n  计算评估指标...")
    metrics = calculate_metrics(
        predictions, classnames, similarity_matrix=similarity_matrix,
        valid_indices=valid_indices, top_k_values=[1, 3, 5, 10]
    )
    
    if metrics:
        print(f"\n{'='*60}")
        print(f"评估指标:")
        print(f"{'='*60}")
        print(f"  Top-k 准确率:")
        for k in [1, 3, 5, 10]:
            if f'acc@{k}' in metrics:
                print(f"    acc@{k}: {metrics[f'acc@{k}']:.4f}")
        
        print(f"\n  F1 分数:")
        print(f"    F1 Micro: {metrics.get('f1_micro', 0):.4f}")
        print(f"    F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"    Precision Micro: {metrics.get('precision_micro', 0):.4f}")
        print(f"    Recall Micro: {metrics.get('recall_micro', 0):.4f}")
        print(f"    Precision Macro: {metrics.get('precision_macro', 0):.4f}")
        print(f"    Recall Macro: {metrics.get('recall_macro', 0):.4f}")
        
        print(f"\n  平均精度均值:")
        print(f"    mAP: {metrics.get('map', 0):.4f}")
        
        print(f"\n  样本统计:")
        print(f"    有效样本（有真实标签）: {metrics.get('valid_samples', 0)}")
        print(f"    总样本数: {metrics.get('total_samples', 0)}")
        print(f"{'='*60}\n")
    else:
        print("  ⚠ 警告: 没有有效的带真实标签的预测，无法计算指标")
    
    # 7. 打印部分预测结果
    print(f"\n{'='*60}")
    print(f"预测结果示例（前5个样本，带真实标签对比）:")
    print(f"{'='*60}")
    for i, pred in enumerate(predictions[:5]):
        print(f"\n样本 {i+1} (video_id: {pred['video_id']}):")
        print(f"  Top-1 预测: {pred['predicted_top1']}")
        
        if pred.get('ground_truth_human_labels'):
            print(f"  真实标签 (human_labels): {', '.join(pred['ground_truth_human_labels'])}")
        
        print(f"  Top-{args.top_k} 预测:")
        for p in pred['top_k_predictions']:
            print(f"    {p['rank']}. {p['classname']} (相似度: {p['similarity']:.4f})")
    print(f"{'='*60}\n")
    
    # 8. 保存结果
    if args.output_json:
        output_file = args.output_json
    else:
        output_file = f"hyperbolic_clap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    save_data = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "parquet_dir": args.parquet_dir,
            "num_samples": args.num_samples,
            "device": str(device),
            "model_path": model_path,
            "projection_checkpoint": args.projection_checkpoint,
            "c": c,
            "temperature": args.temperature
        },
        "model_info": {
            "model_type": "hyperbolic_clap_modelscope",
            "model_path": model_path,
            "projection_checkpoint": args.projection_checkpoint
        },
        "similarity_stats": {
            "min": float(similarity_matrix.min()),
            "max": float(similarity_matrix.max()),
            "mean": float(similarity_matrix.mean()),
            "std": float(similarity_matrix.std())
        },
        "metrics": metrics if metrics else None,
        "predictions": predictions
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

