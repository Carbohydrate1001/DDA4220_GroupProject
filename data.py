from datasets import load_dataset

# 1. 加载数据集
ds = load_dataset("agkphysics/AudioSet", "balanced")

# 2. 保存到指定路径（例如：./my_audioset_data）
ds.save_to_disk("./my_audioset_data")