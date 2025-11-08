import pandas as pd
import numpy as np
import torch
import laion_clap
from tqdm import tqdm 
import os
import librosa 

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

# --- 配置 ---
# 1. data.csv文件路径
CSV_PATH = '/home/nastukoi/main/ESC-50-master/meta/esc50.csv'

# 2. 存放所有.wav音频文件的文件夹路径
AUDIO_BASE_PATH = '/home/nastukoi/main/ESC-50-master/audio' 

# 3. embeddings.pt文件保存路径 
OUTPUT_PT_PATH = '/home/nastukoi/main/test/clap_embeddings_data.pt'

# 4. CLAP模型
print("开始加载CLAP模型")
device = torch.device('cuda')
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt(ckpt="/home/nastukoi/main/laion_clap/630k-audioset-best.pt")
model.to(device)
model.eval()


df = pd.read_csv(CSV_PATH)
all_embeddings = []
all_labels = []

print("开始提取Embedding")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        # 构造完整的文件路径
        file_name = row['filename']
        audio_path = os.path.join(AUDIO_BASE_PATH, file_name)

        with torch.no_grad():
            # 使用 .get_audio_embedding_from_data()
            audio_embed = model.get_audio_embedding_from_filelist(x = [audio_path], use_tensor=True)

        all_embeddings.append(audio_embed) # (512,)
        all_labels.append(row['target'])

    except Exception as e:
        print(f"处理 {audio_path} 时出错: {e}")

try:
    # 1. 把 embedding 列表叠成一个大 tensor
    # 我们的 all_embeddings 是一个 [tensor(1,D), tensor(1,D), ...] 这样的列表
    # 我们用 torch.cat 把它们在 dim=0 (批次维度) 上拼起来
    # 这样就得到了一个 (N, D) 的大 tensor，N 是样本数量，D 是 embedding 维度
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # 2. 把 labels 也变成 tensor，这样更整洁喵~
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 3. 把它们一起放进一个字典里，方便以后读取
    data_to_save = {
        'embeddings': all_embeddings_tensor,
        'labels': all_labels_tensor
    }

    # 4. 定义您的 .pt 文件保存路径 
    OUTPUT_PT_PATH = '/home/nastukoi/main/test/clap_embeddings_data.pt'

    # 5. 保存
    torch.save(data_to_save, OUTPUT_PT_PATH)

    print(f"数据已经保存到 {OUTPUT_PT_PATH} ")
    print(f"保存的 Embeddings 形状: {all_embeddings_tensor.shape}")
    print(f"保存的 Labels 形状: {all_labels_tensor.shape}")

except Exception as e:
    print(f"保存文件时出错了: {e}")