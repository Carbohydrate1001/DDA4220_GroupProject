# CLAP AudioSet 测试脚本使用说明

## 概述

`test_clap_audioset.py` 是一个用于测试 CLAP（Contrastive Language-Audio Pretraining）模型在 AudioSet 数据集上的零样本分类性能的脚本。该脚本遵循官方 CLAP 代码的评估流程，使用预训练模型进行音频分类。

### 主要功能

- ✅ **零样本音频分类**: 无需训练即可对音频进行分类
- ✅ **支持 Parquet 数据格式**: 直接从 parquet 文件读取音频和标签
- ✅ **双模型支持**: 支持 `laion_clap` 和 `ModelScope ClapModel` 两种实现
- ✅ **完整的评估指标**: 提供 Acc@k、mAP、Recall 等多种指标
- ✅ **详细的输出**: 生成包含预测结果和评估指标的 JSON 文件

### 支持的模型

- **laion_clap**: 官方 LAION CLAP 实现（默认）
- **ModelScope ClapModel**: ModelScope 提供的 CLAP 模型实现（推荐，支持 `larger_clap_general` 等更大模型）

## 数据要求

### 1. 数据格式

脚本需要 **Parquet 格式**的数据文件，每个 parquet 文件应包含以下列：

- **`audio`** (必需): 音频数据，格式为字典，包含 `bytes` 字段
  ```python
  {
    'bytes': <音频文件的二进制数据>
  }
  ```

- **`labels`** (可选): 原始标签，可以是字符串或列表
  ```python
  # 示例1: 字符串
  "dog"
  
  # 示例2: 列表
  ["dog", "barking"]
  ```

- **`human_labels`** (推荐): 人类可读的标签，用于生成文本描述和评估
  ```python
  # 示例1: 字符串
  "Dog"
  
  # 示例2: 列表
  ["Dog", "Barking"]
  ```

- **`video_id`** (可选): 视频/音频的唯一标识符

### 2. 数据存放位置

将 parquet 文件放在任意目录下，例如：

```
项目根目录/
├── Datasets/
│   └── AudioSet/
│       ├── audio_0001.parquet
│       ├── audio_0002.parquet
│       └── ...
└── test_clap_audioset.py
```

或者：

```
/path/to/your/data/
├── part1.parquet
├── part2.parquet
└── ...
```

## 环境要求

### 依赖包

根据使用的模型类型，安装相应的依赖包：

#### 使用 laion_clap（默认）

```bash
pip install laion-clap
pip install pandas
pip install librosa
pip install soundfile
pip install torch
pip install numpy
pip install tqdm
```

#### 使用 ModelScope ClapModel

```bash
pip install modelscope
pip install pandas
pip install librosa
pip install soundfile
pip install torch
pip install numpy
pip install tqdm
```

### CLAP 类别标签文件

脚本需要 AudioSet 类别标签文件，应位于：

```
CLAP/class_labels/audioset_class_labels_indices.json
```

如果该文件不存在，分类测试将被跳过。

## 参数说明

### 必需参数

- **`--parquet-dir`**: Parquet 文件所在目录的路径
  ```bash
  --parquet-dir /path/to/parquet/files
  ```

### 可选参数

- **`--num-samples`**: 测试样本数量（默认: 100）
  ```bash
  --num-samples 500
  ```

- **`--audio-dir`**: 音频文件目录（如果 parquet 中包含相对路径，默认: None）
  ```bash
  --audio-dir /path/to/audio/files
  ```

- **`--device`**: 使用的计算设备（默认: `cuda:0`）
  ```bash
  --device cuda:0    # 使用GPU
  --device cpu       # 使用CPU
  ```

- **`--checkpoint`**: CLAP 模型 checkpoint 路径（可选，如果不提供则使用默认checkpoint）
  ```bash
  --checkpoint /path/to/checkpoint.pt
  ```
  
  - 使用 `laion_clap` 时：如果不指定，脚本会自动下载默认的预训练模型
  - 使用 `ModelScope` 时：可以指定本地模型路径（如 `C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general`），如果不指定则使用 `laion/larger_clap_general`

- **`--use-modelscope`**: 使用 ModelScope 的 ClapModel 而不是 laion_clap（默认: False）
  ```bash
  --use-modelscope
  ```

- **`--output-json`**: 输出 JSON 文件路径（可选，默认: `clap_test_results_YYYYMMDD_HHMMSS.json`）
  ```bash
  --output-json results/my_results.json
  ```

## 运行示例

### 基本用法

```bash
# 使用laion_clap（默认，100个样本，GPU，自动下载checkpoint）
python test_clap_audioset.py --parquet-dir ./Datasets/AudioSet

# 使用ModelScope ClapModel（推荐，使用本地模型）
python test_clap_audioset.py --parquet-dir ./Datasets/AudioSet --use-modelscope --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
```

### 完整参数示例

```bash
# 使用laion_clap：测试500个样本，使用自定义checkpoint，指定输出文件
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 500 \
    --device cuda:0 \
    --checkpoint ./checkpoints/clap_model.pt \
    --output-json ./results/test_results.json

# 使用ModelScope：测试500个样本，使用本地模型路径
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 500 \
    --device cuda:0 \
    --use-modelscope \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general" \
    --output-json ./results/test_results.json
```

### CPU 模式

```bash
# 如果只有CPU或GPU不可用
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --device cpu
```

### Windows 系统示例

```powershell
# PowerShell - 使用laion_clap
python test_clap_audioset.py --parquet-dir "E:\Datasets\AudioSet" --num-samples 200

# PowerShell - 使用ModelScope（推荐）
python test_clap_audioset.py --parquet-dir "E:\Datasets\AudioSet" --num-samples 200 --use-modelscope --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"

# CMD - 使用laion_clap
python test_clap_audioset.py --parquet-dir E:\Datasets\AudioSet --num-samples 200

# CMD - 使用ModelScope
python test_clap_audioset.py --parquet-dir E:\Datasets\AudioSet --num-samples 200 --use-modelscope --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
```

## 输出说明

### 1. 控制台输出

脚本运行时会显示详细的进度信息：

```
[1/4] 加载CLAP模型...
  检测到GPU: NVIDIA GeForce RTX 3090 (计算能力: 8.6)
  使用默认checkpoint（将自动下载）
✓
[2/4] 读取parquet文件...
✓ 找到 1 个文件，1000 行/文件
[3/4] 准备测试数据...
 ✓ 100 个样本 (100 个有音频)

[4/4] 运行测试...
  测试1: 音频嵌入提取 (100 个样本)
    提取音频嵌入: 100%|████████████| 100/100 [00:15<00:00,  6.45样本/s]
  测试1: 音频嵌入提取 ✓ (100 个嵌入)

  测试2: 文本嵌入提取 (100 个样本)
    提取文本嵌入: 100%|████████████| 100/100 [00:05<00:00, 18.23文本/s]
  测试2: 文本嵌入提取 ✓ (100 个嵌入)

  测试3: 零样本分类预测
    加载 527 个AudioSet类别...
✓
    计算相似度矩阵...
✓
✓ 完成 100 个样本的分类预测
    计算评估指标...
✓

  评估结果:
    acc@1: 0.4500
    acc@3: 0.6800
    acc@5: 0.7500
    acc@10: 0.8200
    mean_rank: 5.2300
    median_rank: 3.0000
    mAP@10: 0.5234
    recall@1: 0.4500
    recall@5: 0.7500
    recall@10: 0.8200
    valid_samples: 100
    total_samples: 100

  测试4: 音频-文本匹配相似度
✓ 平均相似度: 0.7234 ± 0.1234

保存结果到: clap_test_results_20251130_120000.json...
✓

测试完成!
```

### 2. JSON 输出文件

脚本会生成一个详细的 JSON 结果文件，包含以下结构：

```json
{
  "test_info": {
    "timestamp": "2025-11-30T12:00:00.123456",
    "parquet_dir": "./Datasets/AudioSet",
    "num_samples": 100,
    "audio_dir": null,
    "device": "cuda:0",
    "checkpoint_path": "default",
    "use_modelscope": false
  },
  "model_info": {
    "status": "success",
    "model_type": "laion_clap",
    "enable_fusion": false,
    "device": "cuda:0",
    "checkpoint_path": "default"
  },
  "data_info": {
    "status": "success",
    "num_parquet_files": 1,
    "parquet_files": ["audio_0001.parquet"],
    "columns": ["audio", "labels", "human_labels", "video_id"],
    "total_rows": 1000
  },
  "test_results": {
    "audio_embedding": {
      "status": "success",
      "num_samples": 100,
      "embedding_shape": [100, 512]
    },
    "text_embedding": {
      "status": "success",
      "num_samples": 100,
      "embedding_shape": [100, 512]
    },
    "classification": {
      "status": "success",
      "num_classes": 527,
      "top_k": 5,
      "metrics": {
        "acc@1": 0.45,
        "acc@3": 0.68,
        "acc@5": 0.75,
        "acc@10": 0.82,
        "mean_rank": 5.23,
        "median_rank": 3.0,
        "mAP@10": 0.5234,
        "recall@1": 0.45,
        "recall@5": 0.75,
        "recall@10": 0.82,
        "valid_samples": 100,
        "total_samples": 100
      }
    },
    "similarity": {
      "status": "success",
      "mean": 0.7234,
      "std": 0.1234,
      "max": 0.95,
      "min": 0.45
    }
  },
  "samples": [
    {
      "sample_id": 0,
      "video_id": "video_001",
      "text": "The sounds of Dog, Barking",
      "predicted_labels": [
        {
          "rank": 1,
          "classname": "Dog",
          "similarity": 0.8234
        },
        {
          "rank": 2,
          "classname": "Bark",
          "similarity": 0.7891
        },
        ...
      ],
      "predicted_top1": "Dog",
      "predicted_human_label": "Dog",
      "ground_truth_human_labels": ["Dog", "Barking"],
      "matching_similarity": 0.7234,
      "top_5_text_matches": [...]
    },
    ...
  ]
}
```

### 3. 输出字段说明

#### test_info
- `timestamp`: 测试时间戳
- `parquet_dir`: 输入的 parquet 文件目录
- `num_samples`: 测试样本数量
- `audio_dir`: 音频文件目录（如果指定）
- `device`: 使用的计算设备
- `checkpoint_path`: 使用的 checkpoint 路径
- `use_modelscope`: 是否使用 ModelScope 模型（true/false）

#### model_info
- `status`: 模型加载状态（"success" 或 "failed"）
- `model_type`: 模型类型（"laion_clap" 或 "modelscope"）
- `enable_fusion`: 是否使用融合模型（仅 laion_clap）
- `device`: 使用的计算设备
- `checkpoint_path`: 使用的 checkpoint 路径

#### data_info
- `num_parquet_files`: 找到的 parquet 文件数量
- `parquet_files`: parquet 文件名列表
- `columns`: parquet 文件的列名
- `total_rows`: 总行数

#### test_results.classification.metrics
- `acc@1`: Top-1 准确率（预测的 top-1 类别是否在 ground truth 中）
- `acc@k`: Top-k 准确率（ground truth 是否在 top-k 预测中）
- `mean_rank`: 平均排名（ground truth 在排序中的平均位置）
- `median_rank`: 中位数排名
- `mAP@10`: Mean Average Precision @ 10
- `recall@k`: Recall @ k（等同于 acc@k）

#### samples
每个样本包含：
- `predicted_labels`: Top-k 预测结果（包含排名、类别名、相似度）
- `predicted_top1`: Top-1 预测类别
- `ground_truth_human_labels`: 真实标签
- `matching_similarity`: 音频-文本匹配相似度
- `top_5_text_matches`: 最相似的5个文本匹配

## 评估指标说明

### 准确率指标

- **Acc@1**: 预测的 top-1 类别是否在 ground truth 标签中
- **Acc@k**: Ground truth 标签是否出现在 top-k 预测中
- **Recall@k**: 与 Acc@k 相同，更标准的叫法

### 排名指标

- **Mean Rank**: Ground truth 标签在预测排序中的平均位置（越小越好）
- **Median Rank**: Ground truth 标签在预测排序中的中位数位置

### 精度指标

- **mAP@10**: Mean Average Precision @ 10，综合考虑排名和准确率

## 模型选择说明

脚本支持两种 CLAP 模型实现，可以根据需要选择：

### 1. laion_clap（默认）

**特点**:
- ✅ 官方 LAION 实现，与论文代码完全一致
- ✅ 轻量级，模型文件较小
- ✅ 支持融合和非融合两种模式
- ⚠️ 需要单独下载 checkpoint（首次运行自动下载）

**适用场景**:
- 需要与论文结果对比
- 需要较小的模型文件
- 使用官方训练流程

**使用方式**:
```bash
python test_clap_audioset.py --parquet-dir ./Datasets/AudioSet
```

### 2. ModelScope ClapModel（推荐）

**特点**:
- ✅ 使用 ModelScope 生态，模型管理统一
- ✅ 支持 `larger_clap_general` 等更大更强的模型
- ✅ 可以直接使用本地已下载的模型（无需重新下载）
- ✅ API 更统一，易于集成
- ⚠️ 需要安装 `modelscope` 包
- ⚠️ 模型文件较大

**适用场景**:
- 需要更好的分类性能
- 已经通过 ModelScope 下载了模型
- 需要与 ModelScope 其他模型集成

**使用方式**:
```bash
# 使用本地模型
python test_clap_audioset.py --parquet-dir ./Datasets/AudioSet --use-modelscope --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"

# 自动下载模型
python test_clap_audioset.py --parquet-dir ./Datasets/AudioSet --use-modelscope
```

### 模型对比表

| 特性 | laion_clap | ModelScope |
|------|-----------|------------|
| 模型大小 | 较小 | 较大（larger_clap_general） |
| 性能 | 良好 | 更好 |
| 下载方式 | 自动下载 | 自动下载或使用本地 |
| API 统一性 | 官方API | ModelScope统一API |
| 推荐场景 | 论文复现 | 生产使用 |

**推荐**: 
- 如果你已经从 ModelScope 下载了模型，**推荐使用 ModelScope 模式**
- 如果需要与论文结果对比，使用 `laion_clap` 模式

## 注意事项

1. **音频格式**: 脚本会自动将音频重采样到 48kHz，并处理为单声道
2. **内存使用**: 大量样本可能需要较多内存，建议根据系统配置调整 `--num-samples`
3. **GPU 兼容性**: 如果遇到 CUDA 错误，可以尝试使用 `--device cpu` 或更新 PyTorch 版本
4. **Checkpoint 下载**: 
   - 使用 `laion_clap` 时：首次运行时会自动下载默认 checkpoint（约几百MB），确保网络连接正常
   - 使用 `ModelScope` 时：如果指定本地路径，会直接使用；否则会从 ModelScope 下载
5. **类别标签文件**: 确保 `CLAP/class_labels/audioset_class_labels_indices.json` 文件存在，否则分类测试会被跳过
6. **ModelScope 模型路径**: 
   - Windows: `C:\Users\<用户名>\.cache\modelscope\hub\models\laion\larger_clap_general`
   - Linux/Mac: `~/.cache/modelscope/hub/models/laion/larger_clap_general`
   - 可以通过 `--checkpoint` 参数指定，如果不指定会自动从 ModelScope 下载
7. **音频处理**: 
   - 脚本会自动将音频重采样到 48kHz（ModelScope 和 laion_clap 都要求）
   - 自动处理为单声道
   - 自动归一化到 [-1, 1] 范围
   - 对于过长的音频会截取，过短的音频会填充
8. **批处理**: 脚本会自动进行批处理以提高效率，批大小根据设备自动调整（GPU: 16, CPU: 8）

## 故障排除

### 问题1: CUDA 不可用
```
⚠ 警告: 指定了 cuda:0，但CUDA不可用，切换到CPU
```
**解决**: 使用 `--device cpu` 参数

### 问题2: Checkpoint 下载失败
```
✗ 失败: ConnectionError...
```
**解决**: 
- 检查网络连接
- 手动下载 checkpoint 并使用 `--checkpoint` 参数指定路径
- 或设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 问题3: 未找到类别文件
```
⚠ 未找到类别文件: CLAP/class_labels/audioset_class_labels_indices.json
```
**解决**: 确保在项目根目录运行脚本，或检查 CLAP 目录结构

### 问题4: Parquet 文件格式错误
```
✗ 失败: KeyError: 'audio'
```
**解决**: 检查 parquet 文件是否包含 `audio` 列，且格式正确

### 问题5: ModelScope 模型加载失败
```
✗ 失败: ModuleNotFoundError: No module named 'modelscope'
```
**解决**: 安装 ModelScope: `pip install modelscope`

### 问题6: ModelScope 模型路径找不到
```
⚠ 警告: checkpoint路径不存在
```
**解决**: 
- 检查模型路径是否正确
- Windows 默认路径: `C:\Users\<用户名>\.cache\modelscope\hub\models\laion\larger_clap_general`
- Linux/Mac 默认路径: `~/.cache/modelscope/hub/models/laion/larger_clap_general`
- 如果不指定路径，脚本会自动从 ModelScope 下载模型

## ModelScope 使用示例

### 使用本地 ModelScope 模型

如果你已经从 ModelScope 下载了模型（例如 `larger_clap_general`），可以这样使用：

```bash
# Windows 系统
python test_clap_audioset.py \
    --parquet-dir "E:\Datasets\AudioSet" \
    --num-samples 100 \
    --use-modelscope \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"

# Linux/Mac 系统
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --use-modelscope \
    --checkpoint ~/.cache/modelscope/hub/models/laion/larger_clap_general
```

### 使用 ModelScope 在线模型

如果不指定 `--checkpoint` 参数，脚本会自动从 ModelScope 下载模型：

```bash
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --use-modelscope
```

### ModelScope 模型特点

- **更大的模型**: `larger_clap_general` 比默认的 `laion_clap` 模型更大，性能通常更好
- **统一接口**: 使用 ModelScope 的 `ClapModel` 和 `ClapProcessor`，API 更统一
- **本地缓存**: 模型会自动缓存到 `~/.cache/modelscope/` 或 `C:\Users\<用户名>\.cache\modelscope\` 目录，下次使用更快
- **自动处理**: ModelScope 的 processor 会自动处理音频格式，无需手动预处理

### 查找 ModelScope 模型路径

如果你不确定模型路径，可以：

1. **查看缓存目录**:
   ```bash
   # Windows PowerShell
   ls "C:\Users\$env:USERNAME\.cache\modelscope\hub\models\laion\"
   
   # Linux/Mac
   ls ~/.cache/modelscope/hub/models/laion/
   ```

2. **让脚本自动下载**: 不指定 `--checkpoint` 参数，脚本会自动下载并缓存模型

3. **使用模型ID**: 直接使用模型ID（如 `laion/larger_clap_general`），脚本会自动处理

## 快速开始

### 第一次使用（推荐 ModelScope）

```bash
# 1. 安装依赖
pip install modelscope pandas librosa soundfile torch numpy tqdm

# 2. 准备数据（确保 parquet 文件在指定目录）

# 3. 运行测试（自动下载模型）
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --use-modelscope

# 4. 查看结果
# 结果会保存在 clap_test_results_YYYYMMDD_HHMMSS.json 文件中
```

### 使用已下载的 ModelScope 模型

```bash
# Windows
python test_clap_audioset.py \
    --parquet-dir "E:\Datasets\AudioSet" \
    --use-modelscope \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"

# Linux/Mac
python test_clap_audioset.py \
    --parquet-dir ./Datasets/AudioSet \
    --use-modelscope \
    --checkpoint ~/.cache/modelscope/hub/models/laion/larger_clap_general
```

## 示例数据准备

如果你需要准备测试数据，可以参考以下 Python 代码创建示例 parquet 文件：

```python
import pandas as pd
import soundfile as sf
import numpy as np
import io

# 读取音频文件
audio_data, sr = sf.read('example.wav')
audio_bytes = io.BytesIO()
sf.write(audio_bytes, audio_data, sr, format='WAV')
audio_bytes.seek(0)

# 创建 DataFrame
df = pd.DataFrame({
    'video_id': ['video_001'],
    'audio': [{'bytes': audio_bytes.read()}],
    'labels': [['dog']],
    'human_labels': [['Dog', 'Barking']]
})

# 保存为 parquet
df.to_parquet('example.parquet', engine='pyarrow')
```

## 性能优化建议

1. **使用 GPU**: 如果可用，使用 `--device cuda:0` 可以显著加速
2. **批处理大小**: 脚本会根据设备自动调整批大小，GPU 使用 16，CPU 使用 8
3. **样本数量**: 根据内存情况调整 `--num-samples`，避免内存溢出
4. **模型选择**: ModelScope 的 `larger_clap_general` 性能更好但速度稍慢

## 常见问题 FAQ

### Q: 两种模型方式的结果会一样吗？
A: 不完全一样。ModelScope 的 `larger_clap_general` 是更大的模型，通常性能更好。

### Q: 可以同时使用两种模型吗？
A: 不可以，每次运行只能选择一种模型方式。如果需要对比，可以分别运行两次。

### Q: ModelScope 模型下载很慢怎么办？
A: 可以：
- 使用本地已下载的模型路径
- 检查网络连接
- 使用代理或镜像

### Q: 如何知道模型是否加载成功？
A: 查看控制台输出，如果看到 `✓` 表示加载成功。也可以查看 JSON 结果文件中的 `model_info.status` 字段。

### Q: 音频格式有什么要求？
A: 脚本会自动处理，但建议：
- 采样率: 48kHz（会自动重采样）
- 声道: 单声道（会自动转换）
- 格式: WAV、FLAC 等常见格式

## 联系与支持

如有问题或建议，请参考：
- CLAP 官方仓库: https://github.com/LAION-AI/CLAP
- CLAP 论文: https://arxiv.org/abs/2211.06687
- ModelScope 文档: https://modelscope.cn/docs

