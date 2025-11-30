"""
CLAP ModelScope Zero-shot Classification Prediction Script
Use ModelScope's ClapModel to perform zero-shot classification prediction on audio in parquet files
Only outputs prediction results, does not calculate accuracy
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

from modelscope import ClapModel, ClapProcessor

# 可选的双曲投影支持
try:
    from hyperbolic_projection import HyperbolicProjection, hyperbolic_similarity_matrix_batch
    HYPERBOLIC_AVAILABLE = True
except ImportError:
    HYPERBOLIC_AVAILABLE = False


def load_audioset_classes(class_labels_path):
    """Load AudioSet class labels"""
    if not os.path.exists(class_labels_path):
        print(f"⚠ Warning: Class file does not exist: {class_labels_path}")
        return None
    
    with open(class_labels_path, 'r', encoding='utf-8') as f:
        class_index_dict = json.load(f)
    
    classnames = list(class_index_dict.keys())
    print(f"✓ Loaded {len(classnames)} AudioSet classes")
    return classnames


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
                    # If it's directly an array, use it directly
                    audio = np.array(audio_val['array'], dtype=np.float32)
                    if audio.ndim > 1:
                        # Convert to mono: for shape (samples, channels), take mean along channels dimension
                        if audio.shape[0] > audio.shape[1]:
                            audio = np.mean(audio, axis=1)
                        else:
                            audio = np.mean(audio, axis=0)
                    audio = audio.flatten()
                    # Check audio length
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
                # If it's directly an array
                audio = np.array(audio_val, dtype=np.float32)
                if audio.ndim > 1:
                    # Convert to mono: for shape (samples, channels), take mean along channels dimension
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
                # Try to read with soundfile
                audio_file_obj = io.BytesIO(audio_bytes)
                try:
                    audio, sr = sf.read(audio_file_obj)
                except Exception as sf_error:
                    # If soundfile fails, try librosa
                    try:
                        audio_file_obj.seek(0)  # Reset file pointer
                        audio, sr = librosa.load(audio_file_obj, sr=None)
                    except Exception as librosa_error:
                        sample['audio_error'] = f"soundfile: {sf_error}, librosa: {librosa_error}"
                        sample['has_audio'] = False
                        return sample
                
                # Check audio length
                if len(audio) < 100:
                    sample['audio_error'] = f"Audio too short: {len(audio)}, sample rate: {sr}"
                    sample['has_audio'] = False
                    return sample
                
                # Convert to mono
                # For audio with shape (samples, channels), take mean along channels dimension
                if audio.ndim > 1:
                    if audio.shape[0] > audio.shape[1]:
                        # Shape is (samples, channels), take mean along axis=1
                        audio = np.mean(audio, axis=1)
                    else:
                        # Shape is (channels, samples), take mean along axis=0
                        audio = np.mean(audio, axis=0)
                audio = audio.flatten().astype(np.float32)
                
                # Normalize to [-1, 1]
                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val
                elif max_val == 0:
                    sample['audio_error'] = "Audio is all zeros"
                    sample['has_audio'] = False
                    return sample
                
                # Resample to 48kHz (ModelScope CLAP requirement)
                if sr != 48000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                
                # Final validation: ensure audio data is valid
                if len(audio) < 100:
                    sample['audio_error'] = f"Audio too short after resampling: {len(audio)}"
                    sample['has_audio'] = False
                    return sample
                
                # Ensure audio values are in reasonable range
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
            import traceback
            print(f"  Debug: Error processing audio: {e}")
            print(f"  audio_val type: {type(audio_val)}")
            if isinstance(audio_val, dict):
                print(f"  audio_val keys: {audio_val.keys()}")
    else:
        sample['has_audio'] = False
    
    # Get video_id
    sample['video_id'] = row.get('video_id', '')
    
    # Get ground truth labels
    if 'labels' in row:
        labels = row['labels']
        # Check if labels are valid (handle array and scalar cases)
        if isinstance(labels, (list, np.ndarray)):
            # For arrays, check length and if there are valid values
            if len(labels) > 0:
                # Filter out NaN values
                valid_labels = [label for label in labels if pd.notna(label)]
                sample['labels'] = [str(label) for label in valid_labels] if valid_labels else []
            else:
                sample['labels'] = []
        elif labels is not None and pd.notna(labels):
            # For scalar values
            sample['labels'] = [str(labels)]
        else:
            sample['labels'] = []
    else:
        sample['labels'] = []
    
    if 'human_labels' in row:
        human_labels = row['human_labels']
        # Check if human_labels are valid (handle array and scalar cases)
        if isinstance(human_labels, (list, np.ndarray)):
            # For arrays, check length and if there are valid values
            if len(human_labels) > 0:
                # Filter out NaN values
                valid_labels = [label for label in human_labels if pd.notna(label)]
                sample['human_labels'] = [str(label) for label in valid_labels] if valid_labels else []
            else:
                sample['human_labels'] = []
        elif human_labels is not None and pd.notna(human_labels):
            # For scalar values
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
        print(f"  Extracting audio embeddings and projecting to hyperbolic space ({len(audio_samples)} samples)...")
    else:
        print(f"  Extracting audio embeddings ({len(audio_samples)} samples)...")
    
    # Batch process audio for efficiency and correctness
    for batch_start in tqdm(range(0, len(audio_samples), batch_size), desc="    Processing audio batches", unit="batch", leave=False):
        batch_end = min(batch_start + batch_size, len(audio_samples))
        batch_samples = audio_samples[batch_start:batch_end]
        
        # Prepare batch audio data
        batch_audios = []
        batch_valid_indices = []
        
        for i, sample in enumerate(batch_samples):
            idx = batch_start + i
            if not sample.get('has_audio', False):
                continue
            
            try:
                audio = sample['audio']
                
                # Ensure audio is numpy array
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)
                
                # Ensure it's a 1D array
                if audio.ndim > 1:
                    audio = audio.flatten()
                
                # Validate audio data
                if len(audio) < 100:
                    continue
                
                batch_audios.append(audio)
                batch_valid_indices.append(idx)
                
                # Debug: print statistics for first few audio samples
                if idx < 3:
                    print(f"\n    Sample {idx}: audio length={len(audio)}, sample rate=48000")
                    print(f"      Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}, std={audio.std():.4f}")
                    # Print first few values of audio for comparison
                    print(f"      First 10 audio values: {audio[:10]}")
            except Exception as e:
                print(f"\n    ⚠ Sample {idx} preparation failed: {e}")
                continue
        
        if not batch_audios:
            continue
        
        try:
            # Use ModelScope processor to batch process audio
            # Important: pass list of numpy arrays, sample rate 48kHz
            inputs = processor(audios=batch_audios, sampling_rate=48000, return_tensors="pt").to(device)
            
            with torch.no_grad():
                audio_embed_batch = model.get_audio_features(**inputs)
                
                # Process batch results
                if audio_embed_batch.dim() == 1:
                    # If returned as 1D, batch_size=1
                    audio_embed_batch = audio_embed_batch.unsqueeze(0)
                
                # Validate batch size match
                expected_batch_size = len(batch_audios)
                actual_batch_size = audio_embed_batch.shape[0]
                if expected_batch_size != actual_batch_size:
                    print(f"    ⚠ Warning: Batch size mismatch! Expected: {expected_batch_size}, Actual: {actual_batch_size}")
                
                # Normalize
                audio_embed_batch = torch.nn.functional.normalize(audio_embed_batch, dim=-1)
                
                # Project to hyperbolic space if projection is provided
                if projection is not None:
                    audio_embed_batch = projection(audio_embed_batch)
                
                # Convert to numpy and add to results list
                for j, embed in enumerate(audio_embed_batch):
                    if j >= len(batch_valid_indices):
                        print(f"    ⚠ Warning: Embedding index {j} out of valid index range")
                        break
                    
                    embed_np = embed.cpu().numpy()
                    
                    # Ensure it's a 1D array
                    if embed_np.ndim > 1:
                        embed_np = embed_np.squeeze()
                    
                    audio_embeds.append(embed_np)
                    valid_indices.append(batch_valid_indices[j])
                    
                    # Debug: print statistics for first few embeddings
                    if batch_valid_indices[j] < 3:
                        print(f"      Embedding stats: min={embed_np.min():.4f}, max={embed_np.max():.4f}, mean={embed_np.mean():.4f}, norm={np.linalg.norm(embed_np):.4f}")
                        print(f"      First 5 embedding values: {embed_np[:5]}")
        except Exception as e:
            print(f"\n    ⚠ Batch {batch_start//batch_size} processing failed: {e}")
            import traceback
            traceback.print_exc()
            # If batch processing fails, try processing individually
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
                    
                    inputs = processor(audios=[audio], sampling_rate=48000, return_tensors="pt").to(device)
                    with torch.no_grad():
                        audio_embed = model.get_audio_features(**inputs)
                        if audio_embed.dim() > 1:
                            if audio_embed.shape[0] == 1:
                                audio_embed = audio_embed.squeeze(0)
                            else:
                                audio_embed = audio_embed[0]
                        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
                        embed_np = audio_embed.cpu().numpy()
                        if embed_np.ndim > 1:
                            embed_np = embed_np.squeeze()
                        audio_embeds.append(embed_np)
                        valid_indices.append(idx)
                except Exception as e2:
                    print(f"    ⚠ Sample {idx} individual processing also failed: {e2}")
                    continue
    
    if audio_embeds:
        audio_embeds = np.vstack(audio_embeds)
        print(f"  ✓ Successfully extracted {len(audio_embeds)} audio embeddings, shape: {audio_embeds.shape}")
        
        # Check if audio embeddings are different (debug info)
        if len(audio_embeds) > 1:
            # Calculate differences between first few embeddings
            for i in range(min(3, len(audio_embeds))):
                for j in range(i+1, min(3, len(audio_embeds))):
                    diff = np.abs(audio_embeds[i] - audio_embeds[j]).mean()
                    print(f"  Debug: Average difference between samples {i} and {j} audio embeddings: {diff:.6f}")
                    if diff < 1e-6:
                        print(f"  ⚠ Warning: Samples {i} and {j} audio embeddings are almost identical!")
            
            # Check for duplicate embeddings
            unique_embeds = np.unique(audio_embeds, axis=0)
            if len(unique_embeds) < len(audio_embeds):
                print(f"  ⚠ Warning: Found duplicate audio embeddings! Unique: {len(unique_embeds)}, Total: {len(audio_embeds)}")
        
        return audio_embeds, valid_indices
    else:
        print("  ✗ Failed to extract any audio embeddings")
        return None, []


def extract_text_embeddings(model, processor, texts, device, batch_size=16, projection=None):
    """Extract text embeddings (optionally project to hyperbolic space)"""
    text_embeds = []
    
    use_hyperbolic = projection is not None
    if use_hyperbolic:
        print(f"  Extracting text embeddings and projecting to hyperbolic space ({len(texts)} texts)...")
    else:
        print(f"  Extracting text embeddings ({len(texts)} texts)...")
    pbar = tqdm(total=len(texts), desc="    Processing text", unit="text", leave=False)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                text_embed = model.get_text_features(**inputs)
                # Normalize
                text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
                
                # Project to hyperbolic space if projection is provided
                if projection is not None:
                    text_embed = projection(text_embed)
                
                text_embeds.append(text_embed.cpu().numpy())
        except Exception as e:
            print(f"\n    ⚠ Batch {i//batch_size} processing failed: {e}")
            # Try processing individually
            for text in batch_texts:
                try:
                    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        text_embed = model.get_text_features(**inputs)
                        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
                        text_embeds.append(text_embed.cpu().numpy())
                except:
                    # If individual also fails, use zero vector
                    text_embeds.append(np.zeros((1, text_embeds[0].shape[1] if text_embeds else 512)))
        
        pbar.update(len(batch_texts))
    
    pbar.close()
    
    if text_embeds:
        text_embeds = np.vstack(text_embeds)
        print(f"  ✓ Successfully extracted {len(text_embeds)} text embeddings")
        return text_embeds
    else:
        print("  ✗ Failed to extract any text embeddings")
        return None


def calculate_similarity(audio_embeds, text_embeds, use_hyperbolic=False, c=1.0, temperature=1.0):
    """Calculate similarity between audio and text embeddings"""
    audio_tensor = torch.from_numpy(audio_embeds).float()
    text_tensor = torch.from_numpy(text_embeds).float()
    
    if use_hyperbolic:
        # Use hyperbolic similarity
        similarity = hyperbolic_similarity_matrix_batch(
            audio_tensor, text_tensor, c=c, temperature=temperature
        )
        return similarity.numpy()
    else:
        # Use cosine similarity (original CLAP)
        # Ensure normalization
        audio_tensor = torch.nn.functional.normalize(audio_tensor, dim=-1)
        text_tensor = torch.nn.functional.normalize(text_tensor, dim=-1)
        
        # Calculate similarity matrix (dot product)
        similarity = audio_tensor @ text_tensor.t()
        
        return similarity.numpy()


def calculate_metrics(predictions, classnames, similarity_matrix=None, valid_indices=None, top_k_values=[1, 3, 5, 10]):
    """Calculate evaluation metrics: top-k accuracy, F1 score, and mAP"""
    # Create classname to index mapping
    classname_to_idx = {name: idx for idx, name in enumerate(classnames)}
    
    # Filter predictions with ground truth labels (use human_labels for matching)
    valid_predictions = [p for p in predictions if p.get('ground_truth_human_labels', [])]
    
    if not valid_predictions:
        return None
    
    num_classes = len(classnames)
    num_samples = len(valid_predictions)
    
    # Create mapping from prediction index to valid_predictions index
    pred_idx_to_valid_idx = {}
    valid_idx = 0
    for i, pred in enumerate(predictions):
        if pred.get('ground_truth_human_labels', []):
            pred_idx_to_valid_idx[i] = valid_idx
            valid_idx += 1
    
    # Initialize ground truth and prediction matrices
    y_true = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_pred_topk = {}  # Store predictions for different k values
    
    for k in top_k_values:
        y_pred_topk[k] = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    # Fill ground truth and predictions
    matched_count = 0
    unmatched_labels = []
    
    for i, pred in enumerate(valid_predictions):
        # Ground truth labels - use human_labels for matching with classnames
        gt_labels = pred.get('ground_truth_human_labels', [])
        if not gt_labels:
            # Fallback to labels if human_labels not available
            gt_labels = pred.get('ground_truth_labels', [])
        
        sample_matched = False
        for label in gt_labels:
            label_str = str(label).strip().lower()
            # Try to find matching classname (case-insensitive)
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
    
    # Debug: print matching statistics
    if unmatched_labels:
        unique_unmatched = list(set(unmatched_labels))[:10]  # Show first 10 unique
        print(f"  Debug: {len(unmatched_labels)} ground truth labels could not be matched to classnames")
        print(f"  Debug: Examples of unmatched labels: {unique_unmatched}")
    
    print(f"  Debug: {matched_count}/{num_samples} samples have at least one matched ground truth label")
    
    # Fill top-k predictions
    for i, pred in enumerate(valid_predictions):
        # Top-k predictions
        top_k_preds = pred.get('top_k_predictions', [])
        for k in top_k_values:
            for j in range(min(k, len(top_k_preds))):
                pred_classname = top_k_preds[j]['classname']
                if pred_classname in classname_to_idx:
                    y_pred_topk[k][i, classname_to_idx[pred_classname]] = 1.0
    
    # Calculate metrics
    metrics = {}
    
    # Top-k Accuracy
    for k in top_k_values:
        # For each sample, check if any ground truth label is in top-k predictions
        correct = 0
        for i in range(num_samples):
            # Check if there's any overlap between ground truth and top-k predictions
            if np.any(np.logical_and(y_true[i], y_pred_topk[k][i])):
                correct += 1
        metrics[f'acc@{k}'] = correct / num_samples if num_samples > 0 else 0.0
    
    # Calculate F1 Score (macro and micro)
    # Use top-5 predictions for F1 calculation
    k_for_f1 = min(5, max(top_k_values))
    y_pred_f1 = y_pred_topk[k_for_f1]
    
    # Micro F1: calculate globally
    tp_micro = np.sum(np.logical_and(y_true, y_pred_f1))
    fp_micro = np.sum(np.logical_and(np.logical_not(y_true), y_pred_f1))
    fn_micro = np.sum(np.logical_and(y_true, np.logical_not(y_pred_f1)))
    
    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
    
    metrics['f1_micro'] = float(f1_micro)
    metrics['precision_micro'] = float(precision_micro)
    metrics['recall_micro'] = float(recall_micro)
    
    # Macro F1: average over classes
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
    
    # Calculate mAP (mean Average Precision) using similarity matrix if available
    aps = []
    if similarity_matrix is not None and valid_indices is not None:
        # Create mapping from sample_id to embed_idx
        sample_id_to_embed_idx = {}
        for embed_idx, sample_idx in enumerate(valid_indices):
            sample_id_to_embed_idx[sample_idx] = embed_idx
        
        # Use full similarity matrix for accurate mAP calculation
        for i, pred in enumerate(valid_predictions):
            # Find the corresponding row in similarity matrix
            sample_id = pred.get('sample_id')
            if sample_id is None or sample_id not in sample_id_to_embed_idx:
                continue
            
            embed_idx = sample_id_to_embed_idx[sample_id]
            if embed_idx >= similarity_matrix.shape[0]:
                continue
            
            # Get similarity scores for all classes
            sim_scores = similarity_matrix[embed_idx, :]
            
            # Get ground truth for this sample - use human_labels for matching
            gt_indices = []
            gt_labels = pred.get('ground_truth_human_labels', [])
            if not gt_labels:
                # Fallback to labels if human_labels not available
                gt_labels = pred.get('ground_truth_labels', [])
            
            for label in gt_labels:
                label_str = str(label).strip().lower()
                for classname, idx in classname_to_idx.items():
                    if classname.strip().lower() == label_str:
                        gt_indices.append(idx)
                        break
            
            if not gt_indices:
                continue
            
            # Sort classes by similarity (descending)
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Calculate Average Precision (AP)
            num_relevant = len(gt_indices)
            relevant_found = 0
            precision_at_k = []
            
            for rank, class_idx in enumerate(sorted_indices):
                if class_idx in gt_indices:
                    relevant_found += 1
                    precision_at_k.append(relevant_found / (rank + 1))
            
            # AP is the mean of precision at each relevant item
            ap = np.mean(precision_at_k) if precision_at_k else 0.0
            aps.append(ap)
    else:
        # Fallback: simplified mAP using top-k predictions
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
    
    # Add sample counts
    metrics['valid_samples'] = num_samples
    metrics['total_samples'] = len(predictions)
    
    return metrics


def predict_classification(audio_embeds, class_text_embeds, classnames, test_samples, valid_indices, top_k=10, use_hyperbolic=False, c=1.0, temperature=1.0):
    """Perform zero-shot classification prediction on audio"""
    print(f"\n  Calculating similarity matrix...")
    similarity_matrix = calculate_similarity(audio_embeds, class_text_embeds, use_hyperbolic=use_hyperbolic, c=c, temperature=temperature)
    print(f"  ✓ Similarity matrix shape: {similarity_matrix.shape}")
    
    # Check similarity value range
    sim_min, sim_max = float(similarity_matrix.min()), float(similarity_matrix.max())
    sim_mean = float(similarity_matrix.mean())
    print(f"  Similarity range: [{sim_min:.4f}, {sim_max:.4f}], mean: {sim_mean:.4f}")
    
    if sim_max > 10.0 or sim_min < -1.0:
        print(f"  ⚠ Warning: Similarity value range is abnormal, there may be a problem")
    
    # Generate predictions for each audio
    print(f"\n  Generating classification predictions...")
    predictions = []
    
    # Check if similarity matrix rows are identical (debug)
    if len(similarity_matrix) > 1:
        diff = np.abs(similarity_matrix[0] - similarity_matrix[1]).mean()
        print(f"  Debug: Average difference between first two rows of similarity matrix: {diff:.6f}")
        if diff < 1e-6:
            print(f"  ⚠ Warning: Similarity matrix rows are almost identical, all audio predictions will be the same!")
    
    for embed_idx, sample_idx in enumerate(valid_indices):
        if sample_idx >= len(test_samples):
            continue
        
        sample = test_samples[sample_idx]
        
        # Get similarity between this audio and all classes
        audio_class_sims = similarity_matrix[embed_idx, :]
        ranking = np.argsort(audio_class_sims)[::-1]  # Descending order
        
        # Get top-k predictions
        top_k_indices = ranking[:top_k]
        top_k_predictions = []
        
        for rank, class_idx in enumerate(top_k_indices):
            top_k_predictions.append({
                "rank": rank + 1,
                "classname": classnames[class_idx],
                "similarity": float(audio_class_sims[class_idx])
            })
        
        # Get ground truth labels
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
    
    print(f"  ✓ Generated {len(predictions)} prediction results")
    
    return predictions, similarity_matrix


def main():
    parser = argparse.ArgumentParser(description='CLAP ModelScope Zero-shot Classification Test')
    parser.add_argument('--parquet-dir', type=str, required=True, help='Parquet file directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--checkpoint', type=str, 
                        default=r'C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general',
                        help='ModelScope model path (default uses local cache path)')
    parser.add_argument('--output-json', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--class-labels', type=str, 
                        default='CLAP/class_labels/audioset_class_labels_indices.json',
                        help='AudioSet class labels file path')
    parser.add_argument('--top-k', type=int, default=10, help='Output top-k prediction results, default 10')
    parser.add_argument('--use-hyperbolic', action='store_true', help='Use hyperbolic projection and similarity')
    parser.add_argument('--projection-checkpoint', type=str, default=None, help='Path to hyperbolic projection checkpoint (required if --use-hyperbolic)')
    parser.add_argument('--c', type=float, default=1.0, help='Hyperbolic space curvature parameter (only used with --use-hyperbolic)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for hyperbolic similarity (only used with --use-hyperbolic)')
    
    args = parser.parse_args()
    
    # Validate hyperbolic arguments
    if args.use_hyperbolic:
        if not HYPERBOLIC_AVAILABLE:
            print("⚠ Warning: --use-hyperbolic specified but hyperbolic_projection module not available")
            print("  Falling back to standard CLAP")
            args.use_hyperbolic = False
        elif args.projection_checkpoint is None or not os.path.exists(args.projection_checkpoint):
            print("⚠ Warning: --use-hyperbolic specified but projection checkpoint not found or not specified")
            print(f"  Checkpoint path: {args.projection_checkpoint}")
            print("  Falling back to standard CLAP")
            args.use_hyperbolic = False
    
    # Check device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"⚠ Warning: Specified {args.device}, but CUDA is not available, switching to CPU")
            args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load model
    print("\n[1/4] Loading ModelScope CLAP model...")
    
    # 确定模型路径：优先使用提供的checkpoint，如果不存在则回退到模型ID
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_path = args.checkpoint
        print(f"  Loading from local path: {model_path}")
    else:
        # 如果默认路径不存在，回退到模型ID（ModelScope会自动检查缓存）
        if args.checkpoint:
            print(f"  ⚠ Warning: Specified checkpoint path does not exist: {args.checkpoint}")
            print(f"  Falling back to model ID: laion/larger_clap_general")
        model_path = "laion/larger_clap_general"
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "laion" / "larger_clap_general"
        if cache_dir.exists():
            print(f"  Using model ID: {model_path}")
            print(f"  ✓ Detected local cache, will use directly: {cache_dir}")
        else:
            print(f"  Using model ID: {model_path}")
            print(f"  ℹ ModelScope will automatically check local cache, download if not exists")
    
    try:
        model = ClapModel.from_pretrained(model_path).to(device)
        processor = ClapProcessor.from_pretrained(model_path)
        # Ensure processor knows sample rate (if supported)
        if hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            processor.feature_extractor.sampling_rate = 48000
        model.eval()
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load hyperbolic projection if requested
    projection = None
    if args.use_hyperbolic:
        print("\n[1.5/4] Loading hyperbolic projection layer...")
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
            
            print(f"  ✓ Hyperbolic projection loaded successfully")
            print(f"    Embedding dimension: {embed_dim}")
            print(f"    Curvature parameter: {c}")
            args.c = c  # Use curvature from checkpoint
        except Exception as e:
            print(f"  ✗ Hyperbolic projection loading failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to standard CLAP")
            args.use_hyperbolic = False
            projection = None
    
    # 2. Read parquet files
    print(f"\n[2/4] Reading parquet files...")
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"  ✗ No parquet files found")
        return
    
    print(f"  ✓ Found {len(parquet_files)} parquet files")
    
    # 3. Prepare test data
    print(f"\n[3/4] Preparing test data...")
    test_samples = []
    
    pbar = tqdm(total=min(args.num_samples, 1000), desc="  Reading samples", unit="sample", leave=False)
    
    for parquet_file in parquet_files[:1]:  # Only read first file
        df = pd.read_parquet(parquet_file)
        for idx, row in df.iterrows():
            if len(test_samples) >= args.num_samples:
                break
            
            sample = process_audio_from_parquet(row)
            # Only need audio, ground truth labels not required
            if sample.get('has_audio', False):
                test_samples.append(sample)
                pbar.update(1)
        
        if len(test_samples) >= args.num_samples:
            break
    
    pbar.close()
    
    samples_with_audio = sum(1 for s in test_samples if s.get('has_audio', False))
    print(f"  ✓ Prepared {len(test_samples)} test samples ({samples_with_audio} with audio)")
    
    if not test_samples:
        print("  ✗ No valid test samples found")
        return
    
    # 4. Run test
    print(f"\n[4/4] Running zero-shot classification test...")
    
    # 4.1 Extract audio embeddings
    audio_samples = [s for s in test_samples if s.get('has_audio', False)]
    audio_embeds, valid_indices = extract_audio_embeddings(model, processor, audio_samples, device, projection=projection)
    
    if audio_embeds is None:
        print("  ✗ Unable to extract audio embeddings, test terminated")
        return
    
    # 4.2 Load classes and extract text embeddings
    classnames = load_audioset_classes(args.class_labels)
    if classnames is None:
        print("  ✗ Unable to load class labels, test terminated")
        return
    
    # Generate text prompts for each class
    class_texts = [f"This is a sound of {classname}." for classname in classnames]
    class_text_embeds = extract_text_embeddings(model, processor, class_texts, device, projection=projection)
    
    if class_text_embeds is None:
        print("  ✗ Unable to extract text embeddings, test terminated")
        return
    
    # 4.3 Generate classification predictions
    predictions, similarity_matrix = predict_classification(
        audio_embeds, class_text_embeds, classnames,
        test_samples, valid_indices, top_k=args.top_k,
        use_hyperbolic=args.use_hyperbolic, c=args.c, temperature=args.temperature
    )
    
    if not predictions:
        print("  ✗ Failed to generate prediction results")
        return
    
    # 5. Calculate evaluation metrics
    print(f"\n  Calculating evaluation metrics...")
    metrics = calculate_metrics(predictions, classnames, similarity_matrix=similarity_matrix, 
                               valid_indices=valid_indices, top_k_values=[1, 3, 5, 10])
    
    if metrics:
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics:")
        print(f"{'='*60}")
        print(f"  Top-k Accuracy:")
        for k in [1, 3, 5, 10]:
            if f'acc@{k}' in metrics:
                print(f"    acc@{k}: {metrics[f'acc@{k}']:.4f}")
        
        print(f"\n  F1 Score:")
        print(f"    F1 Micro: {metrics.get('f1_micro', 0):.4f}")
        print(f"    F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"    Precision Micro: {metrics.get('precision_micro', 0):.4f}")
        print(f"    Recall Micro: {metrics.get('recall_micro', 0):.4f}")
        print(f"    Precision Macro: {metrics.get('precision_macro', 0):.4f}")
        print(f"    Recall Macro: {metrics.get('recall_macro', 0):.4f}")
        
        print(f"\n  Mean Average Precision:")
        print(f"    mAP: {metrics.get('map', 0):.4f}")
        
        print(f"\n  Sample Statistics:")
        print(f"    Valid samples (with ground truth): {metrics.get('valid_samples', 0)}")
        print(f"    Total samples: {metrics.get('total_samples', 0)}")
        print(f"{'='*60}\n")
    else:
        print("  ⚠ Warning: No valid predictions with ground truth labels, cannot calculate metrics")
    
    # 6. Print partial prediction results (with ground truth comparison)
    print(f"\n{'='*60}")
    print(f"Prediction Results Examples (first 5 samples, with ground truth comparison):")
    print(f"{'='*60}")
    for i, pred in enumerate(predictions[:5]):
        print(f"\nSample {i+1} (video_id: {pred['video_id']}):")
        print(f"  Top-1 Prediction: {pred['predicted_top1']}")
        
        if pred.get('ground_truth_human_labels'):
            print(f"  Ground Truth (human_labels): {', '.join(pred['ground_truth_human_labels'])}")
        
        print(f"  Top-{args.top_k} Predictions:")
        for p in pred['top_k_predictions']:
            print(f"    {p['rank']}. {p['classname']} (similarity: {p['similarity']:.4f})")
    print(f"{'='*60}\n")
    
    # 7. Save results
    if args.output_json:
        # 如果指定了输出文件，添加序号后缀避免覆盖
        base_name = args.output_json
        if base_name.endswith('.json'):
            base_name = base_name[:-5]  # 移除.json后缀
        
        output_file = f"{base_name}.json"
        counter = 1
        while os.path.exists(output_file):
            output_file = f"{base_name}_{counter}.json"
            counter += 1
    else:
        # 默认文件名使用时间戳
        output_file = f"clap_modelscope_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare data to save
    save_data = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "parquet_dir": args.parquet_dir,
            "num_samples": args.num_samples,
            "device": str(device),
            "model_path": model_path
        },
        "model_info": {
            "model_type": "hyperbolic_clap_modelscope" if args.use_hyperbolic else "modelscope",
            "model_path": model_path,
            "use_hyperbolic": args.use_hyperbolic,
            "projection_checkpoint": args.projection_checkpoint if args.use_hyperbolic else None,
            "c": args.c if args.use_hyperbolic else None,
            "temperature": args.temperature if args.use_hyperbolic else None
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
    
    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

