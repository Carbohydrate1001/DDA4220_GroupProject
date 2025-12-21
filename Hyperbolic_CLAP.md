# Hyperbolic CLAP Training and Inference Guide

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Training Hyperbolic Projector](#training-hyperbolic-projector)
4. [Inference and Testing](#inference-and-testing)
5. [Experimental Results](#experimental-results)
6. [Parameters](#parameters)
7. [Usage Examples](#usage-examples)

---

## Environment Setup

### Python Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU usage)

### Install Dependencies

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Main Dependencies

- **modelscope**: ModelScope library for loading CLAP pretrained models
- **torch**: PyTorch deep learning framework
- **pandas**: Data processing
- **librosa**: Audio processing
- **soundfile**: Audio file I/O
- **tqdm**: Progress bar
- **numpy**: Numerical computation

---

## Data Preparation

### Dataset Structure

Ensure the dataset directory structure is as follows:

```
Datasets/
└── AudioSet/
    ├── train_balanced/     # Training set
    │   ├── 00.parquet
    │   ├── 01.parquet
    │   └── ...
    └── eval/                # Evaluation set
        ├── 00.parquet
        ├── 01.parquet
        └── ...
```

### Parquet File Format

Each parquet file should contain the following columns:

- **`audio`**: Audio data (dictionary format with `bytes` field, or numpy array)
- **`human_labels`**: Human-readable labels (string or list) for training and evaluation
- **`labels`**: Machine-readable labels (optional)
- **`video_id`**: Video/audio ID (optional)

### Class Labels File

Ensure the AudioSet class labels file exists:

```
CLAP/class_labels/audioset_class_labels_indices.json
```

This file contains mappings for 527 AudioSet classes.

---

## Training Hyperbolic Projector

### Basic Training Command

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

### Training Process

The training process includes:

1. **Load CLAP model**: Automatically loads pretrained CLAP model from local cache
2. **Create hyperbolic projection layer**: Initializes a linear projection layer to map CLAP embeddings to hyperbolic space
3. **Load dataset**: Reads audio and label data from parquet files
4. **Data split**: Splits data into training and validation sets (8:2 ratio)
5. **Training loop**: 
   - Freezes CLAP model parameters
   - Trains only the hyperbolic projection layer
   - Optimizes using hyperbolic contrastive loss

### Training Output

Training outputs include:

- Training and validation loss for each epoch
- Positive and negative sample similarities for training and validation sets
- Progress bar showing training progress
- Best model is automatically saved

**Output files**:
- `{output_dir}/projection_epoch_{epoch}.pth`: Checkpoint for each epoch
- `{output_dir}/best_projection_epoch_{epoch}.pth`: Best model (lowest validation loss)
- `{output_dir}/setup.json`: Training log file (contains all parameters and epoch losses)

### Checkpoint File Contents

Each checkpoint file contains:

```python
{
    'epoch': epoch number,
    'projection_state_dict': projection layer parameters,
    'optimizer_state_dict': optimizer state,
    'train_loss': training loss,
    'val_loss': validation loss,
    'c': curvature parameter,
    'temperature': temperature parameter,
    'embed_dim': embedding dimension (usually 512)
}
```

---

## Inference and Testing

### Hyperbolic CLAP Testing

```bash
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_X.pth \
    --output-json results_hyperbolic.json \
    --c 1.0 \
    --temperature 1.0
```

### Original CLAP Testing

```bash
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --output-json results_original.json
```

### Inference Process

1. **Load models**: Loads CLAP model and trained hyperbolic projection layer
2. **Extract embeddings**: 
   - Extracts CLAP embeddings for audio
   - Extracts CLAP embeddings for text (categories)
   - Projects both to hyperbolic space
3. **Compute similarity**: Uses hyperbolic similarity to compute audio-text similarity matrix
4. **Classification**: Performs zero-shot classification based on similarity
5. **Evaluation metrics**: Computes top-k accuracy, F1 score, and mAP

### Output Results

The test generates a JSON file containing:

- **test_info**: Test configuration information
- **model_info**: Model information (whether hyperbolic projection is used)
- **similarity_stats**: Similarity statistics
- **metrics**: Evaluation metrics
  - `acc@1`, `acc@3`, `acc@5`, `acc@10`: Top-k accuracy
  - `f1_micro`, `f1_macro`: F1 scores (micro and macro average)
  - `precision_micro`, `precision_macro`: Precision
  - `recall_micro`, `recall_macro`: Recall
  - `map`: Mean average precision
- **predictions**: Prediction results for each sample

---

## Experimental Results

### Experimental Setup

All experiments were conducted on the AudioSet evaluation set with 500 samples. The experiments compare the performance of original CLAP (using cosine similarity) and hyperbolic CLAP (using Poincaré distance).

### Results Summary

| Model Type | Curvature (c) | Temperature (τ) | Acc@1 | Acc@3 | Acc@5 | Acc@10 | mAP | F1 (micro) | F1 (macro) |
|------------|---------------|-----------------|-------|-------|-------|--------|-----|-----------|-----------|
| Original CLAP (cosine) | - | - | - | - | - | - | - | - | - |
| Hyperbolic CLAP | 0.01 | 1.0 | 36.07% | 57.31% | 66.33% | 79.16% | 38.32% | 28.00% | 28.49% |
| Hyperbolic CLAP | 0.01 | 10.0 | 20.64% | 36.07% | 42.69% | 56.11% | 20.97% | 14.86% | 15.52% |
| Hyperbolic CLAP | 0.1 | 1.0 | 33.27% | 49.90% | 57.11% | 70.34% | 33.15% | 23.23% | 24.57% |

**Notes**: 
- Original CLAP uses cosine similarity with range [-1, 1]
- Hyperbolic CLAP uses Poincaré distance with range (-∞, 0], where larger values (closer to 0) indicate higher similarity
- All experiments use the same test set (500 samples, 499 valid samples)

### Key Findings

1. **Curvature parameter impact**: 
   - Best performance with `c=0.01`, achieving Acc@1 of 36.07% and mAP of 38.32%
   - Performance slightly decreases with `c=0.1`, but still better than temperature=10.0
   - Smaller curvature parameter (c=0.01) provides better representation capability in hyperbolic space

2. **Temperature parameter impact**:
   - Performance is significantly better with `temperature=1.0` than `temperature=10.0`
   - Too large temperature parameter leads to overly smooth similarity distribution, reducing classification performance

3. **Similarity distribution**:
   - Hyperbolic CLAP similarity values are negative (range -54.72 to -36.45 for c=0.01, τ=1.0)
   - Original CLAP similarity values range [-1, 1] with mean near 0
   - Hyperbolic space similarity distribution is more concentrated, beneficial for distinguishing positive and negative samples

### Result Files

All experimental results are saved in the `result/` directory:
- `results_hyperbolic_c0_01.json`: Results for c=0.01, τ=1.0
- `results_hyperbolic_c0_01_t10.json`: Results for c=0.01, τ=10.0
- `results_hyperbolic_c0_1.json`: Results for c=0.1, τ=1.0
- `clap_modelscope_results_*.json`: Original CLAP results

---

## Parameters

### Training Script Parameters (`train_hyperbolic_clap.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--parquet-dir` | str | **Required** | Directory of train_balanced dataset |
| `--num-samples` | int | None | Number of samples to use (None means all) |
| `--device` | str | cuda:0 | Computing device (cuda:0 or cpu) |
| `--checkpoint` | str | Local cache path | ModelScope model path |
| `--output-dir` | str | ./checkpoints_hyperbolic | Output directory |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 16 | Batch size |
| `--learning-rate` | float | 1e-4 | Learning rate |
| `--c` | float | 1.0 | Hyperbolic space curvature parameter |
| `--temperature` | float | 1.0 | Temperature parameter for similarity scaling |
| `--train-ratio` | float | 0.8 | Training set ratio (remainder is validation set) |
| `--seed` | int | 42 | Random seed |
| `--use-swanlab` | flag | False | Enable SwanLab logging |
| `--swanlab-project` | str | Hyperbolic_CLAP | SwanLab project name |
| `--swanlab-workspace` | str | Centauri | SwanLab workspace |

### Testing Script Parameters (`test_clap_modelscope.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--parquet-dir` | str | **Required** | Directory of eval dataset |
| `--projection-checkpoint` | str | **Required** | Path to hyperbolic projection layer checkpoint |
| `--num-samples` | int | 100 | Number of test samples |
| `--device` | str | cuda:0 | Computing device |
| `--checkpoint` | str | Local cache path | ModelScope model path |
| `--output-json` | str | None | Output JSON file path |
| `--class-labels` | str | CLAP/class_labels/... | AudioSet class labels file path |
| `--top-k` | int | 5 | Output top-k predictions |
| `--c` | float | 1.0 | Hyperbolic space curvature parameter (should match training) |
| `--temperature` | float | 1.0 | Temperature parameter (should match training) |
| `--use-hyperbolic` | flag | False | Use hyperbolic projection |

### Hyperbolic Space Parameters

#### Curvature Parameter (`c`)

- **Default**: 1.0
- **Effect**: Controls the curvature of hyperbolic space
- **Impact**: 
  - Smaller `c` means larger curvature, more "curved" hyperbolic space
  - Larger `c` means smaller curvature, closer to Euclidean space
- **Recommendation**: Typically use 1.0, adjust based on experimental results

#### Temperature Parameter (`temperature`)

- **Default**: 1.0
- **Effect**: Scales similarity distribution
- **Impact**:
  - Smaller `temperature` makes similarity distribution sharper (more distinct differences)
  - Larger `temperature` makes similarity distribution smoother
- **Recommendation**: Typically use 1.0, adjust based on experimental results

---

## Usage Examples

### Complete Training Pipeline

```bash
# 1. Train hyperbolic projector (10000 samples, 10 epochs)
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

# 2. Test with trained model
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 1000 \
    --device cuda:0 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_10.pth \
    --output-json results_hyperbolic.json \
    --top-k 5 \
    --c 1.0 \
    --temperature 1.0
```

### Quick Test Example

```bash
# Training (quick test, 1000 samples, 3 epochs)
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --num-samples 1000 \
    --epochs 3 \
    --batch-size 8 \
    --output-dir ./checkpoints_test

# Testing
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_test/best_projection_epoch_3.pth
```

### CPU Mode Example

```bash
# If GPU is unavailable, use CPU (slower)
python train_hyperbolic_clap.py \
    --parquet-dir ./Datasets/AudioSet/train_balanced \
    --device cpu \
    --batch-size 4 \
    --num-samples 1000
```

### Comparison Testing

```bash
# 1. Test original CLAP (cosine similarity)
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --output-json results_original.json

# 2. Test hyperbolic CLAP (hyperbolic similarity)
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet/eval \
    --num-samples 100 \
    --use-hyperbolic \
    --projection-checkpoint ./checkpoints_hyperbolic/best_projection_epoch_10.pth \
    --output-json results_hyperbolic.json
```

---

## Technical Details

### Hyperbolic Projection Layer

This project uses the **Poincaré ball model** to implement hyperbolic space projection. The hyperbolic projection layer consists of two main components:

#### 1. Linear Projection Layer

First, a learnable linear transformation maps CLAP embeddings to the target dimension:

```
x_proj = Linear(x) = W·x + b
```

Where:
- `x`: CLAP embedding [batch_size, 512]
- `W`: Learnable weight matrix [512, 512]
- `b`: Learnable bias vector [512]
- `x_proj`: Linearly projected features [batch_size, 512]

**Initialization**: Weights are initialized using Xavier uniform initialization, bias is initialized to 0.

#### 2. Exponential Map

Maps points from Euclidean space to the Poincaré ball model:

```
y = expmap(x_proj) = (tanh(√c · ||x_proj||) / (√c · ||x_proj||)) · x_proj
```

**Numerical stability**:
- If `||x||` is close to 0, use `tanh(√c·||x||) / (√c·||x||) ≈ 1` to avoid division by zero
- After projection, clip to a ball with radius `clip_r` (default 0.9) to prevent numerical instability

**Output constraint**: Projected points satisfy `||y|| < 1` (inside Poincaré ball)

### Poincaré Distance

In the Poincaré ball model, the hyperbolic distance between two points `x` and `y` is defined as:

```
d_H(x, y) = (1/√c) · arccosh(1 + 2||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
```

**Constraints**: 
- `||x|| < 1` and `||y|| < 1` (must be inside Poincaré ball)
- The argument of `arccosh` must be ≥ 1

### Hyperbolic Similarity

Hyperbolic similarity is defined as the negative of Poincaré distance with temperature scaling:

```
sim(x, y) = -d_H(x, y) / τ
```

Where:
- `τ`: Temperature parameter (default 1.0)
- Similarity range: `(-∞, 0]`, where larger values (closer to 0) indicate higher similarity

### Hyperbolic Contrastive Loss

Training uses **hyperbolic contrastive loss**:

1. **Compute similarity matrix**: For batch size `B` audio-text pairs, compute similarity matrix `S` [B, B]
2. **Compute cross-entropy loss**: 
   - Audio-to-text loss: maximizes diagonal elements
   - Text-to-audio loss: maximizes diagonal elements
   - Total loss: average of both directions

**Optimization objective**: Maximize similarity of positive pairs (matched audio-text), minimize similarity of negative pairs (unmatched audio-text)

### Model Architecture

```
Audio Input → CLAP Audio Encoder → Audio Embedding (512-dim, normalized)
                                          ↓
                                   Linear Projection (trainable)
                                          ↓
                                   Exponential Map
                                          ↓
                                   Hyperbolic Audio Embedding (||h_a|| < 1)

Text Input → CLAP Text Encoder → Text Embedding (512-dim, normalized)
                                          ↓
                                   Linear Projection (trainable)
                                          ↓
                                   Exponential Map
                                          ↓
                                   Hyperbolic Text Embedding (||h_t|| < 1)

Hyperbolic Audio Embedding ↔ Poincaré Distance ↔ Hyperbolic Text Embedding
                                          ↓
                                   Hyperbolic Similarity Matrix S
                                          ↓
                                   Hyperbolic Contrastive Loss L
```

### Training Process

1. **Freeze CLAP parameters**: CLAP encoder parameters remain fixed, only projection layer is trained
2. **Forward pass**: 
   - Extract CLAP embeddings (audio and text)
   - Pass through linear projection layer
   - Apply exponential map to hyperbolic space
3. **Compute loss**: Calculate contrastive loss using hyperbolic similarity matrix
4. **Backward pass**: Update projection layer parameters (W and b)
5. **Optimization objective**: Make matched audio-text pairs closer in hyperbolic space (smaller distance, larger similarity)

---

## References

- **CLAP**: Learning Audio Concepts from Natural Language Supervision
  - Paper: https://arxiv.org/abs/2206.04769
  
- **Poincaré Embeddings**: Learning Hierarchical Representations
  - Paper: https://arxiv.org/abs/1705.08039
  
- **Hyperbolic Neural Networks**
  - Paper: https://arxiv.org/abs/1805.09112
