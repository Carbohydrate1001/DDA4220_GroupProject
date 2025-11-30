"""
查看parquet文件中的标签
"""
import pandas as pd
import numpy as np

# 读取parquet文件
df = pd.read_parquet('Datasets/AudioSet/00.parquet')

print("=" * 60)
print("Parquet文件中的标签信息")
print("=" * 60)
print(f"\n列名: {df.columns.tolist()}")
print(f"\n总行数: {len(df)}")

# 查看前3行的标签
print("\n" + "=" * 60)
print("前3行的标签内容:")
print("=" * 60)

for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"\n行 {i}:")
    print(f"  video_id: {row.get('video_id', 'N/A')}")
    
    # labels列（AudioSet ID）
    if 'labels' in row:
        labels_val = row['labels']
        print(f"  labels (AudioSet ID): {labels_val}")
        print(f"    类型: {type(labels_val)}")
        if isinstance(labels_val, (list, np.ndarray)):
            print(f"    数量: {len(labels_val)}")
            print(f"    内容: {list(labels_val)[:5]}")
    
    # human_labels列（人类可读的标签）
    if 'human_labels' in row:
        human_labels_val = row['human_labels']
        print(f"  human_labels (人类可读): {human_labels_val}")
        print(f"    类型: {type(human_labels_val)}")
        if isinstance(human_labels_val, (list, np.ndarray)):
            print(f"    数量: {len(human_labels_val)}")
            print(f"    内容: {list(human_labels_val)[:5]}")

print("\n" + "=" * 60)
print("说明:")
print("=" * 60)
print("1. labels列: 包含AudioSet的ID（如 '/m/09x0r'），需要映射到类别名")
print("2. human_labels列: 包含人类可读的类别名（如 'Speech', 'Music'）")
print("3. 评估时使用 human_labels 作为ground truth（更直接）")
print("=" * 60)


