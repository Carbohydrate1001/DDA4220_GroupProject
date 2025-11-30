"""
CLAP测试脚本 - 使用AudioSet parquet文件进行测试
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

# 设置HuggingFace镜像站
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import laion_clap

def calculate_evaluation_metrics(test_samples, classnames, class_similarity_np, audio_embeds):
    """计算评估指标：准确率、Top-k准确率、mAP等"""
    metrics = {}
    
    # 创建类别名到索引的映射（不区分大小写）
    classname_to_idx = {}
    classname_lower_to_idx = {}
    for idx, name in enumerate(classnames):
        classname_to_idx[name] = idx
        classname_lower_to_idx[name.lower()] = idx
    
    # 收集ground truth和预测
    ground_truth_indices = []
    predicted_indices = []
    predicted_ranks = []
    valid_samples = 0
    
    for idx in range(len(audio_embeds)):
        if idx >= len(test_samples):
            continue
        
        sample = test_samples[idx]
        
        # 获取ground truth标签（使用human_labels，因为它们是可读的类别名）
        gt_labels = sample.get('ground_truth_human_labels', [])
        if not gt_labels:
            continue
        
        # 将ground truth标签映射到类别索引
        gt_indices = []
        for label in gt_labels:
            label_clean = label.strip()
            # 尝试直接匹配
            if label_clean in classname_to_idx:
                gt_indices.append(classname_to_idx[label_clean])
            elif label_clean.lower() in classname_lower_to_idx:
                gt_indices.append(classname_lower_to_idx[label_clean.lower()])
            else:
                # 尝试模糊匹配（部分匹配）
                for classname, class_idx in classname_to_idx.items():
                    if (label_clean.lower() in classname.lower() or 
                        classname.lower() in label_clean.lower() or
                        label_clean.lower().replace(' ', '') == classname.lower().replace(' ', '')):
                        gt_indices.append(class_idx)
                        break
        
        if not gt_indices:
            continue
        
        # 获取预测（相似度最高的类别）
        audio_class_sims = class_similarity_np[idx, :]
        ranking = np.argsort(audio_class_sims)[::-1]  # 降序排列
        
        predicted_idx = ranking[0]
        predicted_indices.append(predicted_idx)
        
        # 对于多标签情况，检查预测是否匹配任一ground truth标签
        # Top-1准确率：预测的top-1是否在ground truth标签中
        is_correct = predicted_idx in gt_indices
        if is_correct:
            ground_truth_indices.append(predicted_idx)  # 匹配的标签
        else:
            ground_truth_indices.append(gt_indices[0])  # 使用第一个ground truth标签
        
        # 计算真实标签在排序中的最佳位置（rank，从0开始）
        # 找到所有ground truth标签中排名最高的
        best_rank = len(classnames)  # 初始化为最大值
        for gt_idx in gt_indices:
            for rank, pred_idx in enumerate(ranking):
                if pred_idx == gt_idx:
                    if rank < best_rank:
                        best_rank = rank
                    break
        
        predicted_ranks.append(best_rank)
        
        valid_samples += 1
    
    if valid_samples == 0:
        return {"error": "没有有效的ground truth标签", "valid_samples": 0}
    
    predicted_ranks = np.array(predicted_ranks)
    predicted_indices = np.array(predicted_indices)
    ground_truth_indices = np.array(ground_truth_indices)
    
    # 重新计算Acc@1，确保正确处理多标签情况
    # 对于每个样本，检查预测的top-1是否在ground truth标签中
    acc1_correct = 0
    for i, pred_idx in enumerate(predicted_indices):
        # 找到对应的ground truth索引列表
        sample_idx = i
        if sample_idx < len(test_samples):
            sample = test_samples[sample_idx]
            gt_labels = sample.get('ground_truth_human_labels', [])
            if not gt_labels:
                continue
            
            # 将gt_labels映射到类别索引
            gt_indices_list = []
            for label in gt_labels:
                label_clean = label.strip()
                if label_clean in classname_to_idx:
                    gt_indices_list.append(classname_to_idx[label_clean])
                elif label_clean.lower() in classname_lower_to_idx:
                    gt_indices_list.append(classname_lower_to_idx[label_clean.lower()])
            
            # 如果预测在ground truth中，算正确
            if pred_idx in gt_indices_list:
                acc1_correct += 1
    
    # 计算指标
    # 1. Top-1准确率（Acc@1）- 预测的top-1类别是否在ground truth标签中
    metrics["acc@1"] = float(acc1_correct / len(predicted_indices)) if len(predicted_indices) > 0 else 0.0
    
    # 2. Top-k准确率（Acc@k）- ground truth是否在top-k预测中
    for k in [3, 5, 10]:
        acc_k = np.mean(predicted_ranks < k)
        metrics[f"acc@{k}"] = float(acc_k)
    
    # 3. 平均排名（Mean Rank）- ground truth在排序中的平均位置
    metrics["mean_rank"] = float(predicted_ranks.mean() + 1)  # +1因为rank从0开始
    
    # 4. 中位数排名（Median Rank）
    metrics["median_rank"] = float(np.floor(np.median(predicted_ranks)) + 1)
    
    # 5. mAP@10 (Mean Average Precision)
    map_scores = []
    for rank in predicted_ranks:
        if rank < 10:
            map_scores.append(1.0 / (rank + 1))
        else:
            map_scores.append(0.0)
    metrics["mAP@10"] = float(np.mean(map_scores))
    
    # 6. Recall@k (等同于Acc@k，但更标准的叫法)
    for k in [1, 5, 10]:
        recall_k = np.mean(predicted_ranks < k)
        metrics[f"recall@{k}"] = float(recall_k)
    
    metrics["valid_samples"] = valid_samples
    metrics["total_samples"] = len(audio_embeds)
    
    return metrics

def safe_notna(val):
    """安全地检查值是否为NaN，支持数组类型"""
    if isinstance(val, (list, dict, np.ndarray)):
        if isinstance(val, np.ndarray):
            return val.size > 0
        elif isinstance(val, list):
            return len(val) > 0
        elif isinstance(val, dict):
            return len(val) > 0
        return True
    try:
        return pd.notna(val)
    except (ValueError, TypeError):
        return val is not None and str(val) != 'nan' and str(val) != 'NaN'

def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_results_to_json(results, output_file):
    """保存结果到JSON文件"""
    # 转换numpy类型为Python原生类型
    results_serializable = convert_to_serializable(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

def test_clap_with_parquet(parquet_dir, num_samples=100, audio_dir=None, device='cuda:0', output_json=None):
    """使用parquet文件测试CLAP模型"""
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "parquet_dir": parquet_dir,
            "num_samples": num_samples,
            "audio_dir": audio_dir,
            "device": device
        },
        "model_info": {},
        "data_info": {},
        "test_results": {
            "audio_embedding": {},
            "text_embedding": {},
            "similarity": {}
        },
        "samples": []
    }
    
    # 1. 加载CLAP模型
    print("[1/4] 加载CLAP模型...", end=' ', flush=True)
    
    # 检查CUDA
    import torch
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"\n⚠ 警告: 指定了 {device}，但CUDA不可用，切换到CPU")
            device = 'cpu'
        else:
            # 检查GPU计算能力兼容性
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_capability = torch.cuda.get_device_capability(0)
                print(f"\n  检测到GPU: {gpu_name} (计算能力: {gpu_capability[0]}.{gpu_capability[1]})")
                # sm_120需要PyTorch 2.5+或nightly版本
                if gpu_capability[0] >= 12:
                    print(f"  ⚠ 警告: GPU计算能力 {gpu_capability[0]}.{gpu_capability[1]} 可能需要PyTorch 2.5+或nightly版本")
                    print(f"  如果遇到CUDA错误，请尝试:")
                    print(f"    1. 安装nightly版本: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    print(f"    2. 或使用CPU模式: --device cpu")
            except:
                pass
    
    try:
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt()
        print("✓")
        results["model_info"] = {"status": "success", "enable_fusion": False, "device": device}
    except Exception as e:
        print(f"✗ 失败: {e}")
        results["model_info"] = {"status": "failed", "error": str(e)}
        if output_json:
            save_results_to_json(results, output_json)
        return
    
    # 2. 读取parquet文件
    print(f"[2/4] 读取parquet文件...", end=' ', flush=True)
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"✗ 未找到parquet文件")
        results["data_info"]["status"] = "failed"
        results["data_info"]["error"] = f"在 {parquet_dir} 中未找到parquet文件"
        if output_json:
            save_results_to_json(results, output_json)
        return
    
    try:
        df = pd.read_parquet(parquet_files[0])
        print(f"✓ 找到 {len(parquet_files)} 个文件，{len(df)} 行/文件")
        results["data_info"] = {
            "status": "success",
            "num_parquet_files": len(parquet_files),
            "parquet_files": [os.path.basename(f) for f in parquet_files],
            "columns": list(df.columns),
            "total_rows": len(df)
        }
    except Exception as e:
        print(f"✗ 失败: {e}")
        results["data_info"]["status"] = "failed"
        results["data_info"]["error"] = str(e)
        if output_json:
            save_results_to_json(results, output_json)
        return
    
    # 3. 准备测试数据
    print(f"[3/4] 准备测试数据...", end=' ', flush=True)
    df_sample = pd.read_parquet(parquet_files[0])
    
    test_samples = []
    pbar = tqdm(total=min(num_samples, len(df_sample)), desc="准备测试样本", unit="样本", leave=False)
    
    for parquet_file in parquet_files[:1]:
        df = pd.read_parquet(parquet_file)
        for idx, row in df.iterrows():
            if len(test_samples) >= num_samples:
                break
            
            sample = {"row_index": idx, "video_id": row.get('video_id', '')}
            
            # 保存ground truth标签
            if 'labels' in row and safe_notna(row['labels']):
                labels_val = row['labels']
                if isinstance(labels_val, (list, np.ndarray)):
                    sample['ground_truth_labels'] = [str(l) for l in labels_val if safe_notna(l)]
                elif isinstance(labels_val, str):
                    sample['ground_truth_labels'] = [labels_val]
            
            if 'human_labels' in row and safe_notna(row['human_labels']):
                human_labels_val = row['human_labels']
                if isinstance(human_labels_val, (list, np.ndarray)):
                    sample['ground_truth_human_labels'] = [str(l) for l in human_labels_val if safe_notna(l)]
                elif isinstance(human_labels_val, str):
                    sample['ground_truth_human_labels'] = [human_labels_val]
            
            # 从audio列读取bytes
            if 'audio' in row and safe_notna(row['audio']):
                audio_val = row['audio']
                if isinstance(audio_val, dict) and 'bytes' in audio_val:
                    try:
                        audio_bytes = audio_val['bytes']
                        audio_file_obj = io.BytesIO(audio_bytes)
                        audio, sr = sf.read(audio_file_obj)
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1 if audio.shape[0] < audio.shape[1] else 0)
                        audio = audio.flatten().astype(np.float32)
                        max_val = np.abs(audio).max()
                        if max_val > 1.0:
                            audio = audio / max_val
                        if sr != 48000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                        sample['audio_data_processed'] = audio
                        sample['has_audio'] = True
                    except Exception as e:
                        sample['audio_load_error'] = str(e)
            
            # 从human_labels生成文本
            text = ""
            if 'human_labels' in row and safe_notna(row['human_labels']):
                labels_val = row['human_labels']
                if isinstance(labels_val, (list, np.ndarray)):
                    tags = [str(t) for t in labels_val if safe_notna(t) and str(t).strip()]
                    if tags:
                        text = f"The sounds of {', '.join(tags[:5])}"
                        if len(tags) > 5:
                            text += " and more"
                elif isinstance(labels_val, str) and labels_val.strip():
                    text = f"The sound of {labels_val.strip()}"
            
            if text:
                sample['text'] = text
                test_samples.append(sample)
                pbar.update(1)
        
        if len(test_samples) >= num_samples:
            break
    
    pbar.close()
    samples_with_audio = sum(1 for s in test_samples if s.get('has_audio', False))
    print(f" ✓ {len(test_samples)} 个样本 ({samples_with_audio} 个有音频)")
    
    if not test_samples:
        print("✗ 未找到有效的测试样本")
        results["data_info"]["test_samples"] = {"status": "failed", "error": "未找到有效的测试样本"}
        if output_json:
            save_results_to_json(results, output_json)
        return
    
    # 4. 运行测试
    print(f"\n[4/4] 运行测试...")
    
    # 测试1: 音频嵌入提取
    samples_with_audio_data = [s for s in test_samples if s.get('has_audio', False)]
    audio_embeds = None
    
    if samples_with_audio_data:
        print(f"  测试1: 音频嵌入提取 ({len(samples_with_audio_data)} 个样本)")
        try:
            audio_embeds_list = []
            batch_size = 16
            pbar = tqdm(total=len(samples_with_audio_data), desc="    提取音频嵌入", unit="样本", leave=False)
            
            for i in range(0, len(samples_with_audio_data), batch_size):
                batch_samples = samples_with_audio_data[i:i+batch_size]
                processed_audio = []
                
                # 按照CLAP的方式：每个音频都处理到480000长度
                # CLAP的get_audio_features会处理：太长随机截取，太短repeatpad填充
                max_len = 480000  # CLAP的标准长度
                
                for s in batch_samples:
                    if 'audio_data_processed' in s:
                        audio = s['audio_data_processed'].copy()
                        
                        # 模拟CLAP的get_audio_features处理逻辑
                        if len(audio) > max_len:
                            # 随机截取（rand_trunc模式）
                            overflow = len(audio) - max_len
                            idx = np.random.randint(0, overflow + 1)
                            audio = audio[idx:idx + max_len]
                        elif len(audio) < max_len:
                            # repeatpad填充（CLAP的默认方式）
                            n_repeat = int(max_len / len(audio))
                            audio = np.tile(audio, n_repeat)
                            # 剩余部分用零填充
                            remaining = max_len - len(audio)
                            if remaining > 0:
                                audio = np.pad(audio, (0, remaining), mode='constant', constant_values=0)
                        
                        processed_audio.append(audio)
                
                if processed_audio:
                    # 确保所有音频都是480000长度
                    processed_audio = [a[:max_len] if len(a) > max_len else np.pad(a, (0, max_len - len(a)), mode='constant') for a in processed_audio]
                    
                    # stack成(N, T)形状，所有音频都是480000长度
                    batch_audio_array = np.stack(processed_audio)
                    batch_embeds = model.get_audio_embedding_from_data(x=batch_audio_array, use_tensor=False)
                    audio_embeds_list.append(batch_embeds)
                    pbar.update(len(batch_samples))
            
            pbar.close()
            if audio_embeds_list:
                audio_embeds = np.vstack(audio_embeds_list)
                print(f"  测试1: 音频嵌入提取 ✓ ({len(audio_embeds)} 个嵌入)")
                results["test_results"]["audio_embedding"] = {
                    "status": "success",
                    "num_samples": len(audio_embeds),
                    "embedding_shape": list(audio_embeds.shape)
                }
        except Exception as e:
            print(f"  测试1: 音频嵌入提取 ✗ 失败: {e}")
            results["test_results"]["audio_embedding"] = {"status": "failed", "error": str(e)}
    
    # 测试2: 文本嵌入提取
    print(f"\n  测试2: 文本嵌入提取 ({len(test_samples)} 个样本)")
    texts = [s['text'] for s in test_samples]
    text_embeds = None
    
    try:
        print("    初始化tokenizer...", end=' ', flush=True)
        test_embed = model.get_text_embedding([texts[0]], use_tensor=False)
        print("✓")
        
        text_embeds_list = []
        batch_size = 8 if 'CPU' in str(device) else 16
        pbar = tqdm(total=len(texts), desc="    提取文本嵌入", unit="文本", leave=False)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embeds = model.get_text_embedding(batch_texts, use_tensor=False)
                text_embeds_list.append(batch_embeds)
                pbar.update(len(batch_texts))
            except:
                for text in batch_texts:
                    try:
                        single_embed = model.get_text_embedding([text], use_tensor=False)
                        text_embeds_list.append(single_embed)
                        pbar.update(1)
                    except:
                        text_embeds_list.append(np.zeros((1, test_embed.shape[1])))
                        pbar.update(1)
        
        pbar.close()
        if text_embeds_list:
            text_embeds = np.vstack(text_embeds_list)
            print(f"  测试2: 文本嵌入提取 ✓ ({len(text_embeds)} 个嵌入)")
            results["test_results"]["text_embedding"] = {
                "status": "success",
                "num_samples": len(text_embeds),
                "embedding_shape": list(text_embeds.shape)
            }
    except Exception as e:
        print(f"  测试2: 文本嵌入提取 ✗ 失败: {e}")
        results["test_results"]["text_embedding"] = {"status": "failed", "error": str(e)}
    
    # 测试3: 零样本分类（预测标签）
    print(f"\n  测试3: 零样本分类预测", end=' ', flush=True)
    if audio_embeds is not None:
        try:
            # 加载AudioSet类别
            class_labels_path = os.path.join('CLAP', 'class_labels', 'audioset_class_labels_indices.json')
            if os.path.exists(class_labels_path):
                with open(class_labels_path, 'r', encoding='utf-8') as f:
                    class_index_dict = json.load(f)
                classnames = list(class_index_dict.keys())
                
                # 为每个类别生成文本提示
                class_texts = [f"This is a sound of {classname}." for classname in classnames]
                
                print(f"\n    加载 {len(classnames)} 个AudioSet类别...", end=' ', flush=True)
                class_text_embeds = model.get_text_embedding(class_texts, use_tensor=False)
                print("✓")
                
                # 计算每个音频与所有类别的相似度
                print(f"    计算相似度矩阵...", end=' ', flush=True)
                audio_tensor = torch.from_numpy(audio_embeds).float()
                class_text_tensor = torch.from_numpy(class_text_embeds).float()
                audio_tensor = torch.nn.functional.normalize(audio_tensor, dim=-1)
                class_text_tensor = torch.nn.functional.normalize(class_text_tensor, dim=-1)
                
                # 相似度矩阵: (num_audio, num_classes)
                class_similarity = audio_tensor @ class_text_tensor.t()
                class_similarity_np = class_similarity.cpu().numpy()
                print("✓")
                
                # 为每个样本预测top-k类别
                top_k = 5
                for idx in range(len(audio_embeds)):
                    # 获取该音频与所有类别的相似度
                    audio_class_sims = class_similarity_np[idx, :]
                    
                    # 找到top-k类别
                    top_k_indices = np.argsort(audio_class_sims)[-top_k:][::-1]
                    top_k_predictions = []
                    for rank, class_idx in enumerate(top_k_indices):
                        top_k_predictions.append({
                            "rank": rank + 1,
                            "classname": classnames[class_idx],
                            "similarity": float(audio_class_sims[class_idx])
                        })
                    
                    # 获取预测的人类标签（top-1）
                    predicted_human_label = classnames[top_k_indices[0]]
                    
                    # 获取ground truth人类标签
                    gt_human_labels = []
                    if idx < len(test_samples):
                        sample = test_samples[idx]
                        gt_human_labels = sample.get('ground_truth_human_labels', [])
                    
                    # 更新或创建样本结果
                    if idx < len(results["samples"]):
                        results["samples"][idx]["predicted_labels"] = top_k_predictions
                        results["samples"][idx]["predicted_top1"] = predicted_human_label
                        results["samples"][idx]["predicted_human_label"] = predicted_human_label
                        results["samples"][idx]["ground_truth_human_labels"] = gt_human_labels
                    else:
                        sample_result = {
                            "sample_id": idx,
                            "video_id": test_samples[idx].get('video_id', '') if idx < len(test_samples) else '',
                            "predicted_labels": top_k_predictions,
                            "predicted_top1": predicted_human_label,
                            "predicted_human_label": predicted_human_label,
                            "ground_truth_human_labels": gt_human_labels
                        }
                        results["samples"].append(sample_result)
                
                print(f"✓ 完成 {len(audio_embeds)} 个样本的分类预测")
                
                # 计算评估指标
                print(f"    计算评估指标...", end=' ', flush=True)
                evaluation_metrics = calculate_evaluation_metrics(
                    test_samples, classnames, class_similarity_np, audio_embeds
                )
                print("✓")
                
                # 打印评估结果
                print(f"\n  评估结果:")
                for metric_name, metric_value in evaluation_metrics.items():
                    print(f"    {metric_name}: {metric_value:.4f}")
                
                results["test_results"]["classification"] = {
                    "status": "success",
                    "num_classes": len(classnames),
                    "top_k": top_k,
                    "metrics": evaluation_metrics
                }
            else:
                print(f"⚠ 未找到类别文件: {class_labels_path}")
                results["test_results"]["classification"] = {
                    "status": "skipped",
                    "reason": f"类别文件不存在: {class_labels_path}"
                }
        except Exception as e:
            print(f"✗ 失败: {e}")
            results["test_results"]["classification"] = {"status": "failed", "error": str(e)}
    
    # 测试4: 音频-文本匹配相似度计算
    print(f"\n  测试4: 音频-文本匹配相似度", end=' ', flush=True)
    if audio_embeds is not None and text_embeds is not None:
        try:
            min_len = min(len(audio_embeds), len(text_embeds))
            audio_embeds = audio_embeds[:min_len]
            text_embeds = text_embeds[:min_len]
            
            audio_tensor = torch.from_numpy(audio_embeds).float()
            text_tensor = torch.from_numpy(text_embeds).float()
            audio_tensor = torch.nn.functional.normalize(audio_tensor, dim=-1)
            text_tensor = torch.nn.functional.normalize(text_tensor, dim=-1)
            
            similarity = audio_tensor @ text_tensor.t()
            similarity_np = similarity.cpu().numpy()
            diagonal = np.diag(similarity_np)
            
            mean_sim = float(np.mean(diagonal))
            std_sim = float(np.std(diagonal))
            print(f"✓ 平均相似度: {mean_sim:.4f} ± {std_sim:.4f}")
            
            # 保存每个样本的匹配相似度信息
            for idx in range(min_len):
                # 获取ground truth人类标签
                gt_human_labels = []
                if idx < len(test_samples):
                    gt_human_labels = test_samples[idx].get('ground_truth_human_labels', [])
                
                if idx < len(results["samples"]):
                    # 如果已有分类结果，更新匹配信息
                    results["samples"][idx]["text"] = texts[idx]
                    results["samples"][idx]["matching_similarity"] = float(diagonal[idx])
                    
                    # 确保有ground truth标签
                    if "ground_truth_human_labels" not in results["samples"][idx]:
                        results["samples"][idx]["ground_truth_human_labels"] = gt_human_labels
                    
                    # 找到最相似的5个文本（包括自己）
                    text_similarities = similarity_np[idx, :]
                    top_5_indices = np.argsort(text_similarities)[-5:][::-1]
                    results["samples"][idx]["top_5_text_matches"] = []
                    for rank, match_idx in enumerate(top_5_indices):
                        results["samples"][idx]["top_5_text_matches"].append({
                            "rank": rank + 1,
                            "text": texts[match_idx] if match_idx < len(texts) else "",
                            "similarity": float(text_similarities[match_idx]),
                            "is_match": bool(match_idx == idx)
                        })
                else:
                    # 如果没有分类结果，创建新条目
                    sample_result = {
                        "sample_id": idx,
                        "video_id": test_samples[idx].get('video_id', '') if idx < len(test_samples) else '',
                        "text": texts[idx],
                        "matching_similarity": float(diagonal[idx]),
                        "ground_truth_human_labels": gt_human_labels,
                        "top_5_text_matches": []
                    }
                    text_similarities = similarity_np[idx, :]
                    top_5_indices = np.argsort(text_similarities)[-5:][::-1]
                    for rank, match_idx in enumerate(top_5_indices):
                        sample_result["top_5_text_matches"].append({
                            "rank": rank + 1,
                            "text": texts[match_idx] if match_idx < len(texts) else "",
                            "similarity": float(text_similarities[match_idx]),
                            "is_match": bool(match_idx == idx)
                        })
                    results["samples"].append(sample_result)
            
            results["test_results"]["similarity"] = {
                "status": "success",
                "mean": mean_sim,
                "std": std_sim,
                "max": float(np.max(diagonal)),
                "min": float(np.min(diagonal))
            }
        except Exception as e:
            print(f"✗ 失败: {e}")
            results["test_results"]["similarity"] = {"status": "failed", "error": str(e)}
    else:
        print("⚠ 跳过（缺少嵌入）")
        results["test_results"]["similarity"] = {"status": "skipped"}
    
    # 保存结果
    if output_json:
        print(f"\n保存结果到: {output_json}...", end=' ', flush=True)
        save_results_to_json(results, output_json)
        print("✓")
    
    print("\n测试完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLAP AudioSet测试脚本')
    parser.add_argument('--parquet-dir', type=str, required=True, help='parquet文件目录')
    parser.add_argument('--num-samples', type=int, default=100, help='测试样本数量')
    parser.add_argument('--audio-dir', type=str, default=None, help='音频文件目录（如果parquet中包含相对路径）')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备 (cuda:0 或 cpu)')
    parser.add_argument('--output-json', type=str, default=None, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    output_file = args.output_json or f"clap_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    test_clap_with_parquet(
        parquet_dir=args.parquet_dir,
        num_samples=args.num_samples,
        audio_dir=args.audio_dir,
        device=args.device,
        output_json=output_file
    )

