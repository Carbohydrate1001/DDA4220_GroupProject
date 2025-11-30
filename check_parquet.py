"""
检查parquet文件的数据结构
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
import io
import soundfile as sf
import librosa

def check_parquet_file(parquet_path, num_rows=5):
    """检查parquet文件的结构和内容"""
    print(f"\n检查文件: {parquet_path}")
    print("="*60)
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(parquet_path)
        print(f"✓ 成功读取parquet文件")
        print(f"  总行数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return
    
    # 检查每列的数据类型和示例
    print(f"\n列信息:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
        # 检查前几个非空值
        non_null = df[col].dropna()
        if len(non_null) > 0:
            sample = non_null.iloc[0]
            print(f"    示例值类型: {type(sample)}")
            if isinstance(sample, dict):
                print(f"    字典keys: {list(sample.keys())}")
            elif isinstance(sample, (list, np.ndarray)):
                print(f"    列表/数组长度: {len(sample)}")
                if len(sample) > 0:
                    print(f"    第一个元素类型: {type(sample[0])}")
            elif isinstance(sample, bytes):
                print(f"    bytes长度: {len(sample)}")
            else:
                print(f"    示例值: {str(sample)[:100]}")
    
    # 详细检查前几行的audio列
    print(f"\n详细检查前{num_rows}行的audio列:")
    for idx in range(min(num_rows, len(df))):
        row = df.iloc[idx]
        print(f"\n  行 {idx}:")
        
        if 'audio' in row and pd.notna(row['audio']):
            audio_val = row['audio']
            print(f"    audio类型: {type(audio_val)}")
            
            if isinstance(audio_val, dict):
                print(f"    字典keys: {list(audio_val.keys())}")
                for key, val in audio_val.items():
                    print(f"      {key}: 类型={type(val)}, ", end="")
                    if isinstance(val, bytes):
                        print(f"长度={len(val)}")
                        # 尝试读取音频
                        try:
                            audio_file_obj = io.BytesIO(val)
                            audio, sr = sf.read(audio_file_obj)
                            print(f"        ✓ soundfile读取成功: 长度={len(audio)}, 采样率={sr}, 形状={audio.shape}")
                            print(f"        音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
                        except Exception as e:
                            print(f"        ✗ soundfile读取失败: {e}")
                            # 尝试librosa
                            try:
                                audio_file_obj.seek(0)
                                audio, sr = librosa.load(audio_file_obj, sr=None)
                                print(f"        ✓ librosa读取成功: 长度={len(audio)}, 采样率={sr}")
                                print(f"        音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
                            except Exception as e2:
                                print(f"        ✗ librosa读取失败: {e2}")
                    elif isinstance(val, (list, np.ndarray)):
                        val_arr = np.array(val)
                        print(f"长度={len(val_arr)}, 形状={val_arr.shape}, dtype={val_arr.dtype}")
                        if len(val_arr) > 0:
                            print(f"        范围: [{val_arr.min():.4f}, {val_arr.max():.4f}]")
                    else:
                        print(f"值={str(val)[:50]}")
            elif isinstance(audio_val, bytes):
                print(f"    直接是bytes，长度={len(audio_val)}")
                try:
                    audio_file_obj = io.BytesIO(audio_val)
                    audio, sr = sf.read(audio_file_obj)
                    print(f"      ✓ soundfile读取成功: 长度={len(audio)}, 采样率={sr}, 形状={audio.shape}")
                except Exception as e:
                    print(f"      ✗ soundfile读取失败: {e}")
            elif isinstance(audio_val, (list, np.ndarray)):
                audio_arr = np.array(audio_val)
                print(f"    直接是数组，长度={len(audio_arr)}, 形状={audio_arr.shape}, dtype={audio_arr.dtype}")
                if len(audio_arr) > 0:
                    print(f"    范围: [{audio_arr.min():.4f}, {audio_arr.max():.4f}]")
            else:
                print(f"    其他类型: {str(audio_val)[:200]}")
        else:
            print(f"    audio列为空或不存在")
        
        # 检查其他列
        if 'video_id' in row:
            print(f"    video_id: {row['video_id']}")
        if 'human_labels' in row:
            print(f"    human_labels: {row['human_labels']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='检查parquet文件的数据结构')
    parser.add_argument('--parquet-dir', type=str, required=True, help='parquet文件目录')
    parser.add_argument('--num-rows', type=int, default=5, help='检查的行数')
    parser.add_argument('--file-index', type=int, default=0, help='检查第几个文件（从0开始）')
    
    args = parser.parse_args()
    
    # 查找parquet文件
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    if not parquet_files:
        print(f"✗ 未找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 检查指定的文件
    if args.file_index >= len(parquet_files):
        print(f"✗ 文件索引 {args.file_index} 超出范围（共{len(parquet_files)}个文件）")
        return
    
    parquet_file = parquet_files[args.file_index]
    check_parquet_file(parquet_file, args.num_rows)


if __name__ == "__main__":
    main()

