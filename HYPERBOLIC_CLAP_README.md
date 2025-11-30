# 双曲CLAP使用说明

## 概述

本项目实现了在CLAP音频编码器输出后添加双曲投影层，并使用双曲相似度替代cosine相似度进行对比学习。该实现不影响原始CLAP功能，通过参数控制是否启用双曲功能。

## 文件说明

- `hyperbolic_projection.py`: 双曲投影层和双曲相似度计算模块
- `train_hyperbolic_clap.py`: 训练脚本，用于训练双曲投影器
- `test_hyperbolic_clap_modelscope.py`: 测试脚本，专门用于测试双曲CLAP
- `test_clap_modelscope.py`: 原始测试脚本（已添加可选的双曲投影功能）

## 安装依赖

确保已安装以下依赖：

```bash
pip install modelscope
pip install pandas
pip install librosa
pip install soundfile
pip install torch
pip install numpy
pip install tqdm
```

## 使用步骤

### 1. 训练双曲投影器

使用 `train_hyperbolic_clap.py` 训练双曲投影层：

```bash
# 使用默认checkpoint路径（C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general）
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

# 或者指定其他checkpoint路径
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general" \
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

**参数说明：**
- `--parquet-dir`: train_balanced数据集目录（必需）
- `--num-samples`: 使用的样本数量（None表示使用全部）
- `--device`: 计算设备（cuda:0或cpu）
- `--checkpoint`: ModelScope模型路径（可选，默认使用 `C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general`）
- `--output-dir`: 输出目录，保存训练好的投影器
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--learning-rate`: 学习率
- `--c`: 双曲空间曲率参数（默认1.0）
- `--temperature`: 温度参数（默认1.0）
- `--train-ratio`: 训练集比例（默认0.8，即8-2分割）

**训练输出：**
- 每个epoch会保存模型到 `{output_dir}/projection_epoch_{epoch}.pth`
- 最佳模型会保存为 `{output_dir}/best_projection_epoch_{epoch}.pth`
- 训练过程中会显示每个epoch的训练损失和验证损失

### 2. 测试双曲CLAP

#### 方法1：使用专门的测试脚本

```bash
# 使用默认checkpoint路径
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

# 或者指定其他checkpoint路径
python test_hyperbolic_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_X.pth \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general" \
    --num-samples 100 \
    --device cuda:0 \
    --output-json results_hyperbolic.json \
    --class-labels CLAP/class_labels/audioset_class_labels_indices.json \
    --top-k 5 \
    --c 1.0 \
    --temperature 1.0
```

**参数说明：**
- `--parquet-dir`: eval数据集目录（必需）
- `--projection-checkpoint`: 训练好的投影器检查点路径（必需）
- `--num-samples`: 测试样本数量
- `--device`: 计算设备
- `--checkpoint`: ModelScope模型路径（可选，默认使用 `C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general`）
- `--output-json`: 输出JSON文件路径
- `--class-labels`: AudioSet类别标签文件路径
- `--top-k`: 输出top-k预测结果
- `--c`: 双曲空间曲率参数（应与训练时一致）
- `--temperature`: 温度参数（应与训练时一致）

#### 方法2：使用原始测试脚本（带双曲选项）

```bash
# 使用原始CLAP（cosine相似度，使用默认checkpoint路径）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100

# 使用双曲CLAP（双曲相似度，使用默认checkpoint路径）
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_X.pth \
    --c 1.0 \
    --temperature 1.0
```

## 双曲空间说明

### Poincaré球模型

本项目使用Poincaré球模型实现双曲空间投影：

1. **指数映射（expmap）**: 将欧几里得空间的点映射到Poincaré球模型
2. **Poincaré距离**: 计算双曲空间中两点之间的距离
3. **双曲相似度**: 使用Poincaré距离的负值作为相似度度量

### 关键参数

- **曲率参数（c）**: 控制双曲空间的曲率，c越小曲率越大
- **温度参数（temperature）**: 用于缩放相似度，影响softmax分布的尖锐程度
- **裁剪半径（clip_r）**: 防止数值不稳定，默认0.9

## 训练过程说明

1. **数据加载**: 从train_balanced目录读取parquet文件
2. **数据分割**: 按8-2比例分割训练集和验证集
3. **CLAP模型**: 使用ModelScope的预训练CLAP模型，参数冻结
4. **投影层训练**: 只训练双曲投影层，使用双曲对比损失
5. **损失函数**: 使用双曲相似度计算对比损失

## 测试过程说明

1. **加载模型**: 加载CLAP模型和训练好的双曲投影层
2. **提取嵌入**: 提取音频和文本的CLAP嵌入
3. **双曲投影**: 将嵌入投影到双曲空间
4. **计算相似度**: 使用双曲相似度计算音频-文本相似度矩阵
5. **分类预测**: 基于相似度矩阵进行零样本分类
6. **评估指标**: 计算top-k准确率、F1分数和mAP

## 注意事项

1. **曲率参数一致性**: 训练和测试时使用的曲率参数（c）应该一致
2. **温度参数**: 温度参数影响相似度分布，可根据需要调整
3. **内存使用**: 大批量数据可能需要较多内存，可适当减小batch_size
4. **数值稳定性**: 双曲空间计算可能涉及数值稳定性问题，已添加裁剪保护

## 输出文件

### 训练输出

- `{output_dir}/projection_epoch_{epoch}.pth`: 每个epoch的模型检查点
- `{output_dir}/best_projection_epoch_{epoch}.pth`: 最佳模型检查点

检查点包含：
- `projection_state_dict`: 投影层参数
- `optimizer_state_dict`: 优化器状态
- `epoch`: 训练轮数
- `train_loss`: 训练损失
- `val_loss`: 验证损失
- `c`: 曲率参数
- `temperature`: 温度参数
- `embed_dim`: 嵌入维度

### 测试输出

- JSON文件包含：
  - `test_info`: 测试信息（时间戳、数据集路径、参数等）
  - `model_info`: 模型信息（模型类型、路径、双曲参数等）
  - `similarity_stats`: 相似度统计
  - `metrics`: 评估指标（top-k准确率、F1、mAP等）
  - `predictions`: 预测结果

## 示例输出

训练过程示例：

```
使用设备: cuda:0

[1/4] 加载ModelScope CLAP模型...
  ✓ CLAP模型加载成功（参数已冻结）

[2/4] 创建双曲投影层...
  ✓ 双曲投影层创建成功
    输入维度: 512
    输出维度: 512
    曲率参数: 1.0

[3/4] 加载数据集...
  ✓ 找到 38 个parquet文件
  ✓ 加载了 10000 个有效样本
  ✓ 数据集分割完成
    训练集: 8000 样本
    验证集: 2000 样本

[4/4] 设置训练参数...
  ✓ 优化器: Adam (lr=0.0001)
  ✓ 损失函数: 双曲对比损失 (c=1.0, temperature=1.0)

============================================================
开始训练
============================================================

Epoch 1/10
------------------------------------------------------------
  Epoch 1 结果:
    训练损失: 2.3456
    验证损失: 2.1234
    ✓ 保存最佳模型到: ./checkpoints_hyperbolic/best_projection_epoch_1.pth
...
```

测试过程示例：

```
使用设备: cuda:0

[1/5] 加载ModelScope CLAP模型...
  ✓ 模型加载成功

[2/5] 加载双曲投影层...
  ✓ 投影层加载成功
    嵌入维度: 512
    曲率参数: 1.0

[3/5] 读取parquet文件...
  ✓ 找到 35 个parquet文件

[4/5] 准备测试数据...
  ✓ 准备了 100 个测试样本（100 个有音频）

[5/5] 运行零样本分类测试...
  提取音频嵌入并投影到双曲空间 (100 样本)...
  ✓ 成功提取 100 个双曲音频嵌入，形状: (100, 512)
  提取文本嵌入并投影到双曲空间 (527 文本)...
  ✓ 成功提取 527 个双曲文本嵌入
  计算双曲相似度矩阵...
  ✓ 相似度矩阵形状: (100, 527)
  相似度范围: [-5.2341, -0.1234], 均值: -2.3456
  生成分类预测...
  ✓ 生成了 100 个预测结果

  计算评估指标...
  调试: 100/100 个样本至少有一个匹配的真实标签

============================================================
评估指标:
============================================================
  Top-k 准确率:
    acc@1: 0.3200
    acc@3: 0.6100
    acc@5: 0.7200
    acc@10: 0.8400

  F1 分数:
    F1 Micro: 0.2956
    F1 Macro: 0.1256
    ...
```

## 故障排除

### 问题1: 投影层加载失败

**错误**: `投影层检查点不存在` 或 `投影层加载失败`

**解决**: 
- 检查检查点路径是否正确
- 确保检查点文件完整
- 检查嵌入维度是否匹配

### 问题2: 数值不稳定

**错误**: `NaN` 或 `Inf` 值出现在损失中

**解决**:
- 减小学习率
- 调整曲率参数（增大c值）
- 检查数据是否包含异常值

### 问题3: 内存不足

**错误**: `CUDA out of memory`

**解决**:
- 减小batch_size
- 减小num_samples
- 使用CPU模式（--device cpu）

### 问题4: 相似度值异常

**错误**: 相似度值范围异常

**解决**:
- 检查曲率参数是否正确
- 检查投影层是否正确加载
- 验证音频和文本嵌入是否正常

## 参考文献

- CLAP: Learning Audio Concepts from Natural Language Supervision (https://arxiv.org/abs/2206.04769)
- Poincaré Embeddings for Learning Hierarchical Representations (https://arxiv.org/abs/1705.08039)
- Hyperbolic Neural Networks (https://arxiv.org/abs/1805.09112)

