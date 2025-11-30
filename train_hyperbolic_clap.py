"""
双曲CLAP投影器训练脚本
训练一个双曲投影层，将CLAP的音频嵌入投影到双曲空间，并使用双曲相似度计算对比损失
"""
import argparse
import os
import glob
import json
import io
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from datetime import datetime
import random

from modelscope import ClapModel, ClapProcessor
from hyperbolic_projection import HyperbolicProjection, HyperbolicContrastiveLoss

# 可选导入swanlab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("⚠ 警告: swanlab未安装，将跳过swanlab日志记录")


class AudioSetDataset(Dataset):
    """AudioSet数据集类"""
    def __init__(self, parquet_files, num_samples=None, processor=None):
        self.processor = processor
        self.samples = []
        
        print(f"  加载数据...")
        pbar = tqdm(total=min(num_samples or 10000, 10000), desc="    读取样本", unit="样本", leave=False)
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                for idx, row in df.iterrows():
                    if num_samples and len(self.samples) >= num_samples:
                        break
                    
                    sample = self.process_audio_from_parquet(row)
                    if sample.get('has_audio', False) and sample.get('human_labels'):
                        self.samples.append(sample)
                        pbar.update(1)
                
                if num_samples and len(self.samples) >= num_samples:
                    break
            except Exception as e:
                print(f"    ⚠ 读取文件 {parquet_file} 失败: {e}")
                continue
        
        pbar.close()
        print(f"  ✓ 加载了 {len(self.samples)} 个有效样本")
    
    def process_audio_from_parquet(self, row):
        """从parquet行中提取和处理音频数据"""
        sample = {}
        
        # 读取音频字节
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
                            sample['has_audio'] = False
                            return sample
                        sample['audio'] = audio
                        sample['has_audio'] = True
                        sample['sample_rate'] = 48000
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
                        sample['has_audio'] = False
                        return sample
                    sample['audio'] = audio
                    sample['has_audio'] = True
                    sample['sample_rate'] = 48000
                
                if audio_bytes is not None:
                    audio_file_obj = io.BytesIO(audio_bytes)
                    try:
                        audio, sr = sf.read(audio_file_obj)
                    except Exception:
                        try:
                            audio_file_obj.seek(0)
                            audio, sr = librosa.load(audio_file_obj, sr=None)
                        except Exception:
                            sample['has_audio'] = False
                            return sample
                    
                    if len(audio) < 100:
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
                        sample['has_audio'] = False
                        return sample
                    
                    if sr != 48000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                    
                    if len(audio) < 100 or np.abs(audio).max() == 0:
                        sample['has_audio'] = False
                        return sample
                    
                    sample['audio'] = audio
                    sample['has_audio'] = True
                    sample['sample_rate'] = 48000
                else:
                    sample['has_audio'] = False
            except Exception as e:
                sample['has_audio'] = False
        else:
            sample['has_audio'] = False
        
        # 获取标签
        if 'human_labels' in row:
            human_labels = row['human_labels']
            if isinstance(human_labels, (list, np.ndarray)):
                valid_labels = [str(label) for label in human_labels if pd.notna(label)]
                sample['human_labels'] = valid_labels if valid_labels else []
            elif human_labels is not None and pd.notna(human_labels):
                sample['human_labels'] = [str(human_labels)]
            else:
                sample['human_labels'] = []
        else:
            sample['human_labels'] = []
        
        sample['video_id'] = row.get('video_id', '')
        
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio': sample['audio'],
            'text': sample['human_labels'][0] if sample['human_labels'] else '',  # 使用第一个标签作为文本
            'video_id': sample['video_id']
        }


def collate_fn(batch):
    """批处理函数"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    return {
        'audios': audios,
        'texts': texts,
        'video_ids': video_ids
    }


def train_epoch(model, clap_model, clap_processor, projection, criterion, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    projection.train()
    clap_model.eval()  # CLAP模型保持eval模式
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1} 训练", unit="批次", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        audios = batch['audios']
        texts = batch['texts']
        
        # 过滤掉空文本
        valid_indices = [i for i, t in enumerate(texts) if t]
        if not valid_indices:
            continue
        
        audios = [audios[i] for i in valid_indices]
        texts = [texts[i] for i in valid_indices]
        
        if len(audios) == 0:
            continue
        
        try:
            # 提取CLAP音频嵌入（不更新梯度）
            with torch.no_grad():
                inputs = clap_processor(audios=audios, sampling_rate=48000, return_tensors="pt").to(device)
                audio_embeds = clap_model.get_audio_features(**inputs)
                if audio_embeds.dim() == 1:
                    audio_embeds = audio_embeds.unsqueeze(0)
                audio_embeds = F.normalize(audio_embeds, dim=-1)
            
            # 提取CLAP文本嵌入（不更新梯度）
            with torch.no_grad():
                text_inputs = clap_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeds = clap_model.get_text_features(**text_inputs)
                text_embeds = F.normalize(text_embeds, dim=-1)
            
            # 投影到双曲空间
            audio_hyperbolic = projection(audio_embeds)
            text_hyperbolic = projection(text_embeds)
            
            # 计算损失
            labels = torch.arange(len(audio_hyperbolic), device=device)
            loss = criterion(audio_hyperbolic, text_hyperbolic, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"\n    ⚠ 批次 {batch_idx} 处理失败: {e}")
            continue
    
    pbar.close()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def validate_epoch(model, clap_model, clap_processor, projection, criterion, dataloader, device, epoch):
    """验证一个epoch"""
    projection.eval()
    clap_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1} 验证", unit="批次", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            audios = batch['audios']
            texts = batch['texts']
            
            valid_indices = [i for i, t in enumerate(texts) if t]
            if not valid_indices:
                continue
            
            audios = [audios[i] for i in valid_indices]
            texts = [texts[i] for i in valid_indices]
            
            if len(audios) == 0:
                continue
            
            try:
                # 提取CLAP嵌入
                inputs = clap_processor(audios=audios, sampling_rate=48000, return_tensors="pt").to(device)
                audio_embeds = clap_model.get_audio_features(**inputs)
                if audio_embeds.dim() == 1:
                    audio_embeds = audio_embeds.unsqueeze(0)
                audio_embeds = F.normalize(audio_embeds, dim=-1)
                
                text_inputs = clap_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeds = clap_model.get_text_features(**text_inputs)
                text_embeds = F.normalize(text_embeds, dim=-1)
                
                # 投影到双曲空间
                audio_hyperbolic = projection(audio_embeds)
                text_hyperbolic = projection(text_embeds)
                
                # 计算损失
                labels = torch.arange(len(audio_hyperbolic), device=device)
                loss = criterion(audio_hyperbolic, text_hyperbolic, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"\n    ⚠ 验证批次 {batch_idx} 处理失败: {e}")
                continue
    
    pbar.close()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='训练双曲CLAP投影器')
    parser.add_argument('--parquet-dir', type=str, required=True, help='Parquet文件目录（train_balanced）')
    parser.add_argument('--num-samples', type=int, default=None, help='使用的样本数量（None表示使用全部）')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备（cuda:0或cpu）')
    parser.add_argument('--checkpoint', type=str, 
                        default=r'C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general',
                        help='ModelScope模型路径（默认使用本地缓存路径）')
    parser.add_argument('--output-dir', type=str, default='./checkpoints_hyperbolic', help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--c', type=float, default=1.0, help='双曲空间曲率参数')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use-swanlab', action='store_true', help='使用swanlab记录训练过程')
    parser.add_argument('--swanlab-project', type=str, default='Hyperbolic_CLAP', help='SwanLab项目名称')
    parser.add_argument('--swanlab-workspace', type=str, default='Centauri', help='SwanLab工作空间')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 检查设备
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"⚠ 警告: 指定了 {args.device}，但CUDA不可用，切换到CPU")
            args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SwanLab（如果启用）
    swanlab_initialized = False
    if args.use_swanlab:
        if SWANLAB_AVAILABLE:
            try:
                swanlab.init(
                    project=args.swanlab_project,
                    workspace=args.swanlab_workspace,
                    config={
                        "parquet_dir": args.parquet_dir,
                        "num_samples": args.num_samples,
                        "device": str(device),
                        "checkpoint": args.checkpoint,
                        "output_dir": args.output_dir,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "c": args.c,
                        "temperature": args.temperature,
                        "train_ratio": args.train_ratio,
                        "seed": args.seed,
                    }
                )
                swanlab_initialized = True
                print("  ✓ SwanLab初始化成功")
            except Exception as e:
                print(f"  ⚠ SwanLab初始化失败: {e}")
                swanlab_initialized = False
        else:
            print("  ⚠ SwanLab未安装，跳过日志记录")
    
    # 1. 加载CLAP模型
    print("\n[1/4] 加载ModelScope CLAP模型...")
    
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
        clap_model = ClapModel.from_pretrained(model_path).to(device)
        clap_processor = ClapProcessor.from_pretrained(model_path)
        clap_model.eval()
        
        # 冻结CLAP模型参数
        for param in clap_model.parameters():
            param.requires_grad = False
        
        print("  ✓ CLAP模型加载成功（参数已冻结）")
    except Exception as e:
        print(f"  ✗ CLAP模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取CLAP音频嵌入维度
    with torch.no_grad():
        dummy_audio = np.random.randn(48000).astype(np.float32)
        dummy_inputs = clap_processor(audios=[dummy_audio], sampling_rate=48000, return_tensors="pt").to(device)
        dummy_embed = clap_model.get_audio_features(**dummy_inputs)
        embed_dim = dummy_embed.shape[-1]
        print(f"  ✓ CLAP音频嵌入维度: {embed_dim}")
    
    # 2. 创建双曲投影层
    print("\n[2/4] 创建双曲投影层...")
    projection = HyperbolicProjection(
        input_dim=embed_dim,
        output_dim=embed_dim,
        c=args.c,
        clip_r=0.9
    ).to(device)
    
    print(f"  ✓ 双曲投影层创建成功")
    print(f"    输入维度: {embed_dim}")
    print(f"    输出维度: {embed_dim}")
    print(f"    曲率参数: {args.c}")
    
    # 3. 加载数据
    print("\n[3/4] 加载数据集...")
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"  ✗ 未找到parquet文件")
        return
    
    print(f"  ✓ 找到 {len(parquet_files)} 个parquet文件")
    
    dataset = AudioSetDataset(parquet_files, num_samples=args.num_samples, processor=clap_processor)
    
    if len(dataset) == 0:
        print("  ✗ 没有有效样本")
        return
    
    # 分割数据集
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"  ✓ 数据集分割完成")
    print(f"    训练集: {len(train_dataset)} 样本")
    print(f"    验证集: {len(val_dataset)} 样本")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows上使用0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 4. 设置优化器和损失函数
    print("\n[4/4] 设置训练参数...")
    optimizer = optim.Adam(projection.parameters(), lr=args.learning_rate)
    criterion = HyperbolicContrastiveLoss(c=args.c, temperature=args.temperature)
    
    print(f"  ✓ 优化器: Adam (lr={args.learning_rate})")
    print(f"  ✓ 损失函数: 双曲对比损失 (c={args.c}, temperature={args.temperature})")
    
    # 5. 初始化训练记录
    setup_json_path = os.path.join(args.output_dir, 'setup.json')
    
    # 加载或创建setup.json
    if os.path.exists(setup_json_path):
        with open(setup_json_path, 'r', encoding='utf-8') as f:
            setup_data = json.load(f)
    else:
        setup_data = {
            'training_runs': []
        }
    
    # 创建本次训练的记录
    current_run = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'parquet_dir': args.parquet_dir,
            'num_samples': args.num_samples,
            'device': str(device),
            'checkpoint': args.checkpoint,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'c': args.c,
            'temperature': args.temperature,
            'train_ratio': args.train_ratio,
            'seed': args.seed,
            'embed_dim': embed_dim,
        },
        'epochs': []
    }
    
    # 5. 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(
            None, clap_model, clap_processor, projection, criterion,
            train_loader, optimizer, device, epoch
        )
        
        # 验证
        val_loss = validate_epoch(
            None, clap_model, clap_processor, projection, criterion,
            val_loader, device, epoch
        )
        
        # 打印epoch结果
        print(f"  Epoch {epoch+1} 结果:")
        print(f"    训练损失: {train_loss:.4f}")
        print(f"    验证损失: {val_loss:.4f}")
        
        # 记录epoch结果到当前训练记录
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'timestamp': datetime.now().isoformat()
        }
        current_run['epochs'].append(epoch_record)
        
        # 记录到SwanLab
        if swanlab_initialized:
            try:
                swanlab.log({
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "epoch": epoch + 1
                })
            except Exception as e:
                print(f"    ⚠ SwanLab记录失败: {e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, f'best_projection_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'projection_state_dict': projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'c': args.c,
                'temperature': args.temperature,
                'embed_dim': embed_dim,
            }, checkpoint_path)
            print(f"    ✓ 保存最佳模型到: {checkpoint_path}")
        
        # 保存每个epoch的模型
        checkpoint_path = os.path.join(args.output_dir, f'projection_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'projection_state_dict': projection.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'c': args.c,
            'temperature': args.temperature,
            'embed_dim': embed_dim,
        }, checkpoint_path)
    
    # 6. 保存训练记录到setup.json
    # 添加最佳损失和总结信息
    if current_run['epochs']:
        best_epoch_idx = min(range(len(current_run['epochs'])), 
                            key=lambda i: current_run['epochs'][i]['val_loss'])
        current_run['summary'] = {
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(current_run['epochs'][-1]['train_loss']),
            'final_val_loss': float(current_run['epochs'][-1]['val_loss']),
            'total_epochs': args.epochs,
            'best_epoch': best_epoch_idx + 1,
            'best_epoch_train_loss': float(current_run['epochs'][best_epoch_idx]['train_loss']),
            'best_epoch_val_loss': float(current_run['epochs'][best_epoch_idx]['val_loss'])
        }
    else:
        current_run['summary'] = {
            'best_val_loss': float(best_val_loss),
            'total_epochs': args.epochs
        }
    
    # 添加到训练记录列表
    setup_data['training_runs'].append(current_run)
    
    # 保存到文件
    with open(setup_json_path, 'w', encoding='utf-8') as f:
        json.dump(setup_data, f, indent=2, ensure_ascii=False)
    
    # 记录最终结果到SwanLab
    if swanlab_initialized:
        try:
            swanlab.log({
                "best_val_loss": float(best_val_loss),
                "final_train_loss": float(current_run['epochs'][-1]['train_loss']) if current_run['epochs'] else None,
                "final_val_loss": float(current_run['epochs'][-1]['val_loss']) if current_run['epochs'] else None,
            })
            swanlab.finish()
            print("  ✓ SwanLab记录完成")
        except Exception as e:
            print(f"  ⚠ SwanLab完成记录失败: {e}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")
    print(f"训练记录已保存到: {setup_json_path}")


if __name__ == "__main__":
    main()

