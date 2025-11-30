# CLAP ModelScope Zero-shot Classification Test Script

## Overview

`test_clap_modelscope.py` is a simplified script specifically designed for zero-shot audio classification using **ModelScope's ClapModel**.

## Features

- ✅ **ModelScope Specific**: Only uses ModelScope's API, code is simpler
- ✅ **Parquet Data Support**: Reads audio data directly from parquet files
- ✅ **Zero-shot Classification**: Classify audio without training
- ✅ **Prediction Output**: Outputs prediction results with ground truth labels for comparison

## Requirements

```bash
pip install modelscope
pip install pandas
pip install librosa
pip install soundfile
pip install torch
pip install numpy
pip install tqdm
```

## Usage

### Basic Usage

```bash
# Use default model (auto-download)
python test_clap_modelscope.py --parquet-dir ./Datasets/AudioSet --num-samples 100

# Use local model
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
```

### Full Parameters

```bash
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 200 \
    --device cuda:0 \
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general" \
    --output-json results.json \
    --class-labels CLAP/class_labels/audioset_class_labels_indices.json \
    --top-k 5
```

## Parameters

- `--parquet-dir` (required): Parquet file directory
- `--num-samples` (optional): Number of test samples, default 100
- `--device` (optional): Computing device, default `cuda:0`
- `--checkpoint` (optional): ModelScope model path, default uses `laion/larger_clap_general`
- `--output-json` (optional): Output JSON file path
- `--class-labels` (optional): AudioSet class labels file path, default `CLAP/class_labels/audioset_class_labels_indices.json`
- `--top-k` (optional): Output top-k prediction results, default 5

## Data Format Requirements

Parquet files should contain the following columns:

- `audio`: Dictionary format, containing `bytes` field (audio binary data)
- `human_labels`: Human-readable labels (string or list)
- `labels`: Machine-readable labels (optional, stored but not displayed)
- `video_id`: Video/audio ID (optional)

## Output Description

### Console Output

```
Using device: cuda:0

[1/4] Loading ModelScope CLAP model...
  Using model: laion/larger_clap_general (will download automatically)
  ✓ Model loaded successfully

[2/4] Reading parquet files...
  ✓ Found 38 parquet files

[3/4] Preparing test data...
  ✓ Prepared 100 test samples (100 with audio)

[4/4] Running zero-shot classification test...
  Extracting audio embeddings (100 samples)...
  ✓ Successfully extracted 100 audio embeddings, shape: (100, 512)
  ✓ Loaded 527 AudioSet classes
  Extracting text embeddings (527 texts)...
  ✓ Successfully extracted 527 text embeddings

  Calculating similarity matrix...
  ✓ Similarity matrix shape: (100, 527)
  Similarity range: [-0.1234, 0.9876], mean: 0.2345

  Generating classification predictions...
  ✓ Generated 100 prediction results

============================================================
Prediction Results Examples (first 5 samples, with ground truth comparison):
============================================================

Sample 1 (video_id: --PJHxphWEs):
  Top-1 Prediction: Speech
  Ground Truth (human_labels): Speech, Gush
  Top-5 Predictions:
    1. Speech (similarity: 0.8234)
    2. Music (similarity: 0.7123)
    3. Gush (similarity: 0.6891)
    4. Human voice (similarity: 0.6543)
    5. Conversation (similarity: 0.6234)

Sample 2 (video_id: --aE2O5G5WE):
  Top-1 Prediction: Music
  Ground Truth (human_labels): Goat, Music, Speech
  Top-5 Predictions:
    1. Music (similarity: 0.8765)
    2. Musical instrument (similarity: 0.7890)
    3. Speech (similarity: 0.7456)
    4. Singing (similarity: 0.7123)
    5. Goat (similarity: 0.6789)

...

============================================================

Results saved to: clap_modelscope_results_20251130_120000.json
```

### JSON Output

```json
{
  "test_info": {
    "timestamp": "2025-11-30T12:00:00.123456",
    "parquet_dir": "./Datasets/AudioSet",
    "num_samples": 100,
    "device": "cuda:0",
    "model_path": "laion/larger_clap_general"
  },
  "model_info": {
    "model_type": "modelscope",
    "model_path": "laion/larger_clap_general"
  },
  "similarity_stats": {
    "min": -0.1234,
    "max": 0.9876,
    "mean": 0.2345,
    "std": 0.1234
  },
  "predictions": [
    {
      "sample_id": 0,
      "video_id": "--PJHxphWEs",
      "predicted_top1": "Speech",
      "top_k_predictions": [
        {
          "rank": 1,
          "classname": "Speech",
          "similarity": 0.8234
        },
        {
          "rank": 2,
          "classname": "Music",
          "similarity": 0.7123
        },
        ...
      ],
      "ground_truth_labels": ["Speech", "Gush"],
      "ground_truth_human_labels": ["Speech", "Gush"]
    },
    ...
  ]
}
```

## Differences from test_clap_audioset.py

| Feature | test_clap_modelscope.py | test_clap_audioset.py |
|---------|-------------------------|----------------------|
| Model Support | ModelScope only | ModelScope + laion_clap |
| Code Complexity | Simple, ModelScope-specific | Complex, supports two models |
| Functionality | Zero-shot classification | Zero-shot classification + audio-text matching |
| Accuracy Calculation | No (only outputs predictions) | Yes (calculates accuracy metrics) |
| Recommended Use | Only need ModelScope | Need to compare two models |

## Troubleshooting

### Issue 1: Model Download Failed
```
✗ Model loading failed: ConnectionError...
```
**Solution**: Check network connection, or use `--checkpoint` to specify local model path

### Issue 2: CUDA Not Available
```
⚠ Warning: Specified cuda:0, but CUDA is not available, switching to CPU
```
**Solution**: Use `--device cpu` parameter

### Issue 3: Abnormal Similarity Values
```
⚠ Warning: Similarity value range is abnormal, there may be a problem
```
**Solution**: 
- Check if model is loaded correctly
- Verify audio data format is correct
- Check if class labels file exists

### Issue 4: Audio Embeddings Are Identical
```
⚠ Warning: Samples X and Y audio embeddings are almost identical!
```
**Solution**: 
- Check if audio data is being processed correctly
- Verify audio preprocessing (resampling, mono conversion)
- Check if batch processing is working correctly

## Examples

### Windows System

```powershell
# PowerShell
python test_clap_modelscope.py `
    --parquet-dir "E:\Datasets\AudioSet" `
    --num-samples 100 `
    --checkpoint "C:\Users\Centauri\.cache\modelscope\hub\models\laion\larger_clap_general"
```

### Linux/Mac System

```bash
python test_clap_modelscope.py \
    --parquet-dir ./Datasets/AudioSet \
    --num-samples 100 \
    --checkpoint ~/.cache/modelscope/hub/models/laion/larger_clap_general
```

## Notes

1. **Audio Format**: The script automatically resamples audio to 48kHz and converts to mono
2. **Memory Usage**: Large number of samples may require more memory, adjust `--num-samples` according to system configuration
3. **Model Path**: If `--checkpoint` is not specified, the script will automatically download the model from ModelScope
4. **Class Labels**: Ensure the class labels file exists, otherwise classification test cannot proceed
5. **Ground Truth**: The script displays `human_labels` in console output, but stores both `labels` and `human_labels` in JSON output
6. **No Accuracy Calculation**: This script only outputs prediction results and ground truth labels for manual comparison, it does not calculate accuracy metrics
