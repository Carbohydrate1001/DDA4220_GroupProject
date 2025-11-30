# 双曲CLAP训练与推理使用手册

## 目录

1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
3. [训练双曲投影器](#训练双曲投影器)
4. [推理测试](#推理测试)
5. [参数详解](#参数详解)
6. [常见问题](#常见问题)
7. [示例命令](#示例命令)

---

## 环境配置

### 1. Python环境要求

- Python 3.8+
- CUDA 11.8+ (如果使用GPU)

### 2. 安装依赖

所有必需的依赖包都在 `requirements.txt` 中。安装命令：

```bash
pip install -r requirements.txt
```

### 3. 主要依赖包说明

- **modelscope**: ModelScope模型库，用于加载CLAP预训练模型
- **torch**: PyTorch深度学习框架
- **pandas**: 数据处理
- **librosa**: 音频处理
- **soundfile**: 音频文件读写
- **tqdm**: 进度条显示
- **numpy**: 数值计算

### 4. 验证环境

运行以下命令验证环境是否正确安装：

```bash
python -c "import torch; import modelscope; print('✓ 环境配置正确')"
```

---

## 数据准备

### 1. 数据集结构

确保数据集目录结构如下：

```
Datasets/
└── AudioSet/
    ├── train_balanced/     # 训练集（用于训练双曲投影器）
    │   ├── 00.parquet
    │   ├── 01.parquet
    │   └── ...
    └── eval/                # 测试集（用于评估）
        ├── 00.parquet
        ├── 01.parquet
        └── ...
```

### 2. Parquet文件格式要求

每个parquet文件应包含以下列：

- **`audio`**: 音频数据（字典格式，包含`bytes`字段，或直接为numpy数组）
- **`human_labels`**: 人类可读的标签（字符串或列表），用于训练和评估
- **`labels`**: 机器可读的标签（可选）
- **`video_id`**: 视频/音频ID（可选）

**重要**: `human_labels`必须包含与AudioSet类别名称匹配的标签（例如："Speech", "Music", "Dog"等）

### 3. 类别标签文件

确保存在AudioSet类别标签文件：

```
CLAP/class_labels/audioset_class_labels_indices.json
```

该文件包含527个AudioSet类别的映射关系。

---

## 训练双曲投影器

### 1. 基本训练命令

```bash
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --num-samples 10000 \
    --device cuda:0 \
    --output-dir ./checkpoints_hyperbolic \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --c 1.0 \
    --temperature 1.0 \
    --train-ratio 0.8
```

### 2. 训练过程说明

训练过程分为以下步骤：

1. **加载CLAP模型**: 自动从本地缓存加载预训练的CLAP模型（`C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general`）
2. **创建双曲投影层**: 初始化一个线性投影层，将CLAP嵌入投影到双曲空间
3. **加载数据集**: 从parquet文件读取音频和标签数据
4. **数据分割**: 按8:2比例分割训练集和验证集
5. **训练循环**: 
   - 冻结CLAP模型参数（不更新）
   - 只训练双曲投影层
   - 使用双曲对比损失进行优化

### 3. 训练输出

训练过程中会显示：

- 每个epoch的训练损失和验证损失
- 进度条显示训练进度
- 最佳模型会自动保存

**输出文件**:
- `{output_dir}/projection_epoch_{epoch}.pth`: 每个epoch的检查点
- `{output_dir}/best_projection_epoch_{epoch}.pth`: 最佳模型（验证损失最低）

### 4. 检查点文件内容

每个检查点文件包含：

```python
{
    'epoch': 训练轮数,
    'projection_state_dict': 投影层参数,
    'optimizer_state_dict': 优化器状态,
    'train_loss': 训练损失,
    'val_loss': 验证损失,
    'c': 曲率参数,
    'temperature': 温度参数,
    'embed_dim': 嵌入维度（通常是512）
}
```

---

## 推理测试

### 方法1: 使用专门的测试脚本（推荐）

```bash
python test_hyperbolic_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_X.pth \
    --num-samples 100 \
    --device cuda:0 \
    --output-json results_hyperbolic.json \
    --class-labels CLAP/class_labels/audioset_class_labels_indices.json \
    --top-k 5 \
    --c 1.0 \
    --temperature 1.0
```

### 方法2: 使用原始测试脚本（带双曲选项）

```bash
# 原始CLAP（cosine相似度）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100

# 双曲CLAP（双曲相似度）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_X.pth \
    --c 1.0 \
    --temperature 1.0
```

### 推理过程说明

1. **加载模型**: 加载CLAP模型和训练好的双曲投影层
2. **提取嵌入**: 
   - 提取音频的CLAP嵌入
   - 提取文本（类别）的CLAP嵌入
   - 将两者投影到双曲空间
3. **计算相似度**: 使用双曲相似度计算音频-文本相似度矩阵
4. **分类预测**: 基于相似度进行零样本分类
5. **评估指标**: 计算top-k准确率、F1分数和mAP

### 输出结果

测试会生成JSON文件，包含：

- **test_info**: 测试配置信息
- **model_info**: 模型信息（是否使用双曲投影）
- **similarity_stats**: 相似度统计
- **metrics**: 评估指标
  - `acc@1`, `acc@3`, `acc@5`, `acc@10`: Top-k准确率
  - `f1_micro`, `f1_macro`: F1分数（微平均和宏平均）
  - `precision_micro`, `precision_macro`: 精确率
  - `recall_micro`, `recall_macro`: 召回率
  - `map`: 平均精度均值
- **predictions**: 每个样本的预测结果

---

## 参数详解

### 训练脚本参数 (`train_hyperbolic_clap.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--parquet-dir` | str | **必需** | train_balanced数据集目录 |
| `--num-samples` | int | None | 使用的样本数量（None表示使用全部） |
| `--device` | str | cuda:0 | 计算设备（cuda:0或cpu） |
| `--checkpoint` | str | 本地缓存路径 | ModelScope模型路径 |
| `--output-dir` | str | ./checkpoints_hyperbolic | 输出目录 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 16 | 批次大小 |
| `--learning-rate` | float | 1e-4 | 学习率 |
| `--c` | float | 1.0 | 双曲空间曲率参数（越小曲率越大） |
| `--temperature` | float | 1.0 | 温度参数（用于缩放相似度） |
| `--train-ratio` | float | 0.8 | 训练集比例（剩余为验证集） |
| `--seed` | int | 42 | 随机种子 |

### 测试脚本参数 (`test_hyperbolic_clap_modelscope.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--parquet-dir` | str | **必需** | eval数据集目录 |
| `--projection-checkpoint` | str | **必需** | 双曲投影层检查点路径 |
| `--num-samples` | int | 100 | 测试样本数量 |
| `--device` | str | cuda:0 | 计算设备 |
| `--checkpoint` | str | 本地缓存路径 | ModelScope模型路径 |
| `--output-json` | str | None | 输出JSON文件路径 |
| `--class-labels` | str | CLAP/class_labels/... | AudioSet类别标签文件路径 |
| `--top-k` | int | 5 | 输出top-k预测结果 |
| `--c` | float | 1.0 | 双曲空间曲率参数（应与训练时一致） |
| `--temperature` | float | 1.0 | 温度参数（应与训练时一致） |

### 双曲空间参数说明

#### 曲率参数 (`c`)

- **默认值**: 1.0
- **作用**: 控制双曲空间的曲率
- **影响**: 
  - `c`越小，曲率越大，双曲空间越"弯曲"
  - `c`越大，曲率越小，接近欧几里得空间
- **建议**: 通常使用1.0，可根据实验结果调整

#### 温度参数 (`temperature`)

- **默认值**: 1.0
- **作用**: 缩放相似度分布
- **影响**:
  - `temperature`越小，相似度分布越尖锐（差异更明显）
  - `temperature`越大，相似度分布越平滑
- **建议**: 通常使用1.0，可根据实验结果调整

---

## 常见问题

### Q1: 训练时出现CUDA内存不足

**问题**: `CUDA out of memory`

**解决方案**:
1. 减小`--batch-size`（例如从16改为8或4）
2. 减小`--num-samples`（使用更少的训练样本）
3. 使用CPU模式：`--device cpu`（速度较慢）

### Q2: 训练损失不下降

**可能原因**:
1. 学习率过大或过小
2. 数据质量问题
3. 双曲参数设置不当

**解决方案**:
1. 尝试调整`--learning-rate`（例如1e-5或1e-3）
2. 检查数据中的`human_labels`是否正确
3. 尝试不同的`--c`和`--temperature`值

### Q3: 测试时找不到投影层检查点

**问题**: `投影层检查点不存在`

**解决方案**:
1. 检查`--projection-checkpoint`路径是否正确
2. 确保训练已完成并生成了检查点文件
3. 使用绝对路径而不是相对路径

### Q4: 相似度值异常（NaN或Inf）

**问题**: 相似度矩阵包含异常值

**解决方案**:
1. 检查双曲参数`--c`和`--temperature`是否与训练时一致
2. 检查投影层是否正确加载
3. 尝试增大`--c`值（例如2.0）以提高数值稳定性

### Q5: 所有预测结果相同

**问题**: 相似度矩阵的行几乎相同

**解决方案**:
1. 检查音频数据是否正确加载
2. 检查CLAP模型是否正确加载
3. 检查投影层是否正确训练

### Q6: 评估指标全为0

**问题**: 无法匹配真实标签

**解决方案**:
1. 检查`human_labels`是否包含正确的类别名称
2. 确保类别名称与AudioSet类别标签文件中的名称匹配（大小写不敏感）
3. 查看调试输出中的"未匹配标签示例"

### Q7: Windows路径问题

**问题**: 路径中包含反斜杠导致错误

**解决方案**:
- 使用原始字符串：`r"C:\Users\..."` 
- 或使用正斜杠：`"C:/Users/..."`
- 或使用双反斜杠：`"C:\\Users\\..."`

---

## 示例命令

### 完整训练流程示例

```bash
# 1. 训练双曲投影器（使用10000个样本，训练10个epoch）
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --num-samples 10000 \
    --device cuda:0 \
    --output-dir ./checkpoints_hyperbolic \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --c 1.0 \
    --temperature 1.0 \
    --train-ratio 0.8

# 2. 使用训练好的模型进行测试
python test_hyperbolic_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_10.pth \
    --num-samples 1000 \
    --device cuda:0 \
    --output-json results_hyperbolic.json \
    --top-k 5 \
    --c 1.0 \
    --temperature 1.0
```

### 快速测试示例（小规模）

```bash
# 训练（快速测试，使用1000个样本，3个epoch）
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --num-samples 1000 \
    --epochs 3 \
    --batch-size 8 \
    --output-dir ./checkpoints_test

# 测试
python test_hyperbolic_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --projection-checkpoint ./checkpoints_test/best_projection_epoch_3.pth \
    --num-samples 100
```

### CPU模式示例

```bash
# 如果GPU不可用，使用CPU（速度较慢）
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --device cpu \
    --batch-size 4 \
    --num-samples 1000
```

### 对比测试示例

```bash
# 1. 测试原始CLAP（cosine相似度）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --output-json results_original.json

# 2. 测试双曲CLAP（双曲相似度）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_10.pth \
    --output-json results_hyperbolic.json

# 3. 对比两个结果文件
```

---

## 性能优化建议

### 1. 训练优化

- **批次大小**: 根据GPU内存调整，通常16-32效果较好
- **学习率**: 从1e-4开始，如果损失不下降可尝试1e-5
- **数据量**: 建议至少使用10000个样本进行训练
- **训练轮数**: 通常10-20个epoch足够，观察验证损失不再下降即可停止

### 2. 推理优化

- **批次大小**: 可以设置较大（32-64）以提高速度
- **样本数量**: 根据需求调整，建议至少100个样本进行评估
- **GPU使用**: 确保使用GPU以加速推理

### 3. 内存优化

- 如果内存不足，减小批次大小
- 使用`--num-samples`限制数据量
- 考虑使用梯度累积（需要修改代码）

---

## 技术细节

### 双曲空间投影

本项目使用**Poincaré球模型**实现双曲空间投影：

1. **指数映射（expmap）**: 将欧几里得空间的点映射到Poincaré球模型
2. **Poincaré距离**: 计算双曲空间中两点之间的距离
3. **双曲相似度**: 使用Poincaré距离的负值作为相似度度量

### 对比损失

训练使用**双曲对比损失**：
- 将音频和文本嵌入投影到双曲空间
- 使用双曲相似度计算对比损失
- 优化投影层参数，使匹配的音频-文本对在双曲空间中更接近

### 模型架构

```
音频输入 → CLAP音频编码器 → 音频嵌入（512维）
                                    ↓
                            双曲投影层（可训练）
                                    ↓
                            双曲音频嵌入

文本输入 → CLAP文本编码器 → 文本嵌入（512维）
                                    ↓
                            双曲投影层（可训练）
                                    ↓
                            双曲文本嵌入

双曲音频嵌入 ↔ 双曲相似度 ↔ 双曲文本嵌入
```

---

## 参考文献

- **CLAP**: Learning Audio Concepts from Natural Language Supervision
  - Paper: https://arxiv.org/abs/2206.04769
  
- **Poincaré Embeddings**: Learning Hierarchical Representations
  - Paper: https://arxiv.org/abs/1705.08039
  
- **Hyperbolic Neural Networks**
  - Paper: https://arxiv.org/abs/1805.09112

---

## 支持与反馈

如有问题或建议，请检查：
1. 本文档的"常见问题"部分
2. 代码中的注释和文档字符串
3. 运行时的错误信息和调试输出

---

**版本**: 1.0  
**最后更新**: 2025年

