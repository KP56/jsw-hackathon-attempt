# Segmentation Model Evaluation Guide

Complete guide for evaluating the segmentation model using IoU and Dice Coefficient metrics with COCO format annotations.

## Overview

The `evaluate_segmentation.py` script:
- Loads COCO format annotations (polygon-based)
- Runs segmentation model on each image
- Calculates IoU and Dice Coefficient
- Generates comparison visualizations
- Provides detailed metrics report

## Metrics Explained

### IoU (Intersection over Union)
Also known as Jaccard Index:
```
IoU = |Intersection| / |Union|
IoU = TP / (TP + FP + FN)
```
- Range: 0.0 to 1.0
- 1.0 = perfect overlap
- Common threshold: 0.5 for "good" segmentation

### Dice Coefficient
Also known as F1 Score:
```
Dice = 2 * |Intersection| / (|A| + |B|)
Dice = 2 * TP / (2 * TP + FP + FN)
```
- Range: 0.0 to 1.0
- 1.0 = perfect overlap
- More weight on true positives than IoU

### Precision & Recall
```
Precision = TP / (TP + FP)  # How many predicted pixels are correct
Recall = TP / (TP + FN)     # How many ground truth pixels are found
```

## Quick Start

### Prerequisites

Your data should be organized as:
```
images/
├── random_0001.jpg
├── random_0002.jpg
├── ...
└── result.json          # COCO format annotations
```

### Run Evaluation

**Basic usage:**
```bash
python evaluate_segmentation.py
```

**Custom paths:**
```bash
python evaluate_segmentation.py \
    --images-dir ./images \
    --annotations ./images/result.json \
    --output-dir ./evaluation_results
```

**Without visualizations (faster):**
```bash
python evaluate_segmentation.py --no-visualizations
```

### View Results

After running, check:
```bash
# Summary metrics
cat evaluation_results/evaluation_results.json

# Visualizations (comparison images)
ls evaluation_results/visualizations/
```

## Output

### Console Output

```
================================================================================
Segmentation Model Evaluation
================================================================================

1. Loading configuration and model...
   Device: cuda:0
✓ Classifier loaded successfully!
✓ Segmentation model loaded successfully!

2. Loading COCO annotations...
Loaded COCO annotations:
  - Images: 62
  - Annotations: 62
  - Categories: 1

3. Evaluating images...
Processing: 100%|████████████████████| 62/62 [00:45<00:00,  1.37it/s]

4. Calculating aggregate metrics...

================================================================================
EVALUATION RESULTS
================================================================================

Dataset: 62 images

Mean IoU (Intersection over Union): 0.7543 ± 0.1234
Mean Dice Coefficient (F1 Score):   0.8234 ± 0.0987
Mean Precision:                      0.8567
Mean Recall:                         0.8123

IoU Distribution:
  Min:     0.4523
  25th %:  0.6789
  Median:  0.7654
  75th %:  0.8456
  Max:     0.9234

Dice Distribution:
  Min:     0.5234
  25th %:  0.7567
  Median:  0.8345
  75th %:  0.8976
  Max:     0.9567

Detailed results saved to: evaluation_results/evaluation_results.json
Visualizations saved to: evaluation_results/visualizations
================================================================================
```

### JSON Results

`evaluation_results/evaluation_results.json`:

```json
{
  "summary": {
    "num_images": 62,
    "mean_iou": 0.7543,
    "std_iou": 0.1234,
    "mean_dice": 0.8234,
    "std_dice": 0.0987,
    "mean_precision": 0.8567,
    "mean_recall": 0.8123,
    "min_iou": 0.4523,
    "max_iou": 0.9234,
    "median_iou": 0.7654,
    "min_dice": 0.5234,
    "max_dice": 0.9567,
    "median_dice": 0.8345
  },
  "per_image_results": [
    {
      "image_id": 0,
      "file_name": "random_0001.jpg",
      "iou": 0.8123,
      "dice": 0.8956,
      "precision": 0.8734,
      "recall": 0.9234,
      "gt_pixels": 156789,
      "pred_pixels": 148234
    }
  ]
}
```

### Visualization Images

Comparison images show 3 panels:
1. **Left:** Ground truth (green overlay)
2. **Middle:** Prediction (blue overlay)
3. **Right:** Comparison
   - Green: True Positive (correct)
   - Red: False Positive (predicted but not in GT)
   - Yellow: False Negative (in GT but not predicted)

Saved to: `evaluation_results/visualizations/`

## COCO Format Requirements

The JSON file must follow COCO format:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "images/random_0001.jpg",
      "width": 1280,
      "height": 960
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [[x1, y1, x2, y2, ..., xn, yn]],
      "bbox": [x, y, width, height],
      "area": 123456.78
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "segment"
    }
  ]
}
```

**Segmentation formats supported:**
- **Polygon:** `[[x1, y1, x2, y2, ...]]` (list of coordinates)
- **Multiple polygons:** `[[poly1], [poly2], ...]`

## Interpretation

### IoU Scores

| IoU Range | Quality |
|-----------|---------|
| 0.9 - 1.0 | Excellent |
| 0.7 - 0.9 | Good |
| 0.5 - 0.7 | Acceptable |
| 0.3 - 0.5 | Poor |
| 0.0 - 0.3 | Very Poor |

### Dice Scores

| Dice Range | Quality |
|------------|---------|
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Acceptable |
| 0.5 - 0.7 | Poor |
| 0.0 - 0.5 | Very Poor |

### Precision vs Recall

- **High Precision, Low Recall:** Model is conservative (misses some areas)
- **Low Precision, High Recall:** Model is aggressive (over-segments)
- **Both High:** Ideal situation
- **Both Low:** Model needs improvement

## Advanced Usage

### Analyze Specific Images

```python
import json

# Load results
with open('evaluation_results/evaluation_results.json') as f:
    data = json.load(f)

# Find worst performers
results = data['per_image_results']
sorted_by_iou = sorted(results, key=lambda x: x['iou'])

print("Worst 5 images:")
for r in sorted_by_iou[:5]:
    print(f"  {r['file_name']}: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}")
```

### Compare with Baseline

```python
import json
import numpy as np

# Load your results
with open('evaluation_results/evaluation_results.json') as f:
    data = json.load(f)

mean_iou = data['summary']['mean_iou']
mean_dice = data['summary']['mean_dice']

# Compare with baseline
baseline_iou = 0.65
baseline_dice = 0.75

improvement_iou = (mean_iou - baseline_iou) / baseline_iou * 100
improvement_dice = (mean_dice - baseline_dice) / baseline_dice * 100

print(f"IoU improvement: {improvement_iou:+.2f}%")
print(f"Dice improvement: {improvement_dice:+.2f}%")
```

### Export to CSV

```python
import json
import pandas as pd

# Load results
with open('evaluation_results/evaluation_results.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['per_image_results'])

# Save to CSV
df.to_csv('evaluation_results.csv', index=False)

# Calculate statistics
print(df[['iou', 'dice', 'precision', 'recall']].describe())
```

## Troubleshooting

### Issue: Import Error

**Error:** `No module named 'tqdm'`

**Solution:**
```bash
pip install tqdm
```

### Issue: Polygon Error

**Error:** `Error processing image: polygon coordinates invalid`

**Solution:**
- Check COCO JSON format
- Ensure coordinates are within image bounds
- Verify polygon has at least 3 points

### Issue: Model Not Found

**Error:** `No checkpoint found in ./checkpoints/segmentation/`

**Solution:**
```bash
# Check checkpoint exists
ls -la checkpoints/segmentation/

# Verify config.yml paths
cat config.yml | grep checkpoint_dir
```

### Issue: Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```python
# Edit config.yml to use CPU
device:
  use_cuda: false

# Or process fewer images at once
```

### Issue: Visualization Not Saved

**Problem:** No images in visualizations folder

**Solution:**
- Visualizations are saved for first 10 images and any with IoU < 0.5
- Use `--no-visualizations` flag if not needed
- Check disk space

## Performance

### Processing Speed

| Hardware | Resolution | Images/sec |
|----------|-----------|------------|
| GPU (CUDA) | 1280x960 | ~2-3 |
| CPU | 1280x960 | ~0.3-0.5 |

### Memory Usage

- GPU: ~2-4 GB VRAM
- CPU: ~1-2 GB RAM
- Per image: ~10-20 MB

## Best Practices

1. **Run on validation set first**
   - Don't evaluate on training data
   - Use held-out test set for final metrics

2. **Check visualizations**
   - Always inspect visual results
   - Look for systematic errors
   - Verify ground truth quality

3. **Analyze distribution**
   - Don't rely only on mean
   - Check min/max and percentiles
   - Look for outliers

4. **Compare consistently**
   - Use same dataset for comparisons
   - Same preprocessing pipeline
   - Same evaluation metrics

5. **Document results**
   - Save evaluation settings
   - Note model checkpoint used
   - Record date and dataset version

## Integration with Training

### After Training

```bash
# 1. Train model
python -m src.train_segmentation

# 2. Evaluate
python evaluate_segmentation.py

# 3. Compare with previous
python compare_evaluations.py prev_results.json new_results.json
```

### Track Metrics Over Time

```python
import json
from datetime import datetime

# Load current results
with open('evaluation_results/evaluation_results.json') as f:
    current = json.load(f)

# Save with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_file = f'evaluation_history/eval_{timestamp}.json'

with open(history_file, 'w') as f:
    json.dump(current, f)
```

## Examples

### Example 1: Quick Evaluation

```bash
# Run evaluation
python evaluate_segmentation.py

# Check results
cat evaluation_results/evaluation_results.json | grep mean_iou
```

### Example 2: Production Validation

```bash
# Evaluate without visualizations (faster)
python evaluate_segmentation.py --no-visualizations

# Export key metrics
python -c "
import json
with open('evaluation_results/evaluation_results.json') as f:
    data = json.load(f)
    print(f\"IoU: {data['summary']['mean_iou']:.4f}\")
    print(f\"Dice: {data['summary']['mean_dice']:.4f}\")
"
```

### Example 3: Detailed Analysis

```bash
# Run with visualizations
python evaluate_segmentation.py

# Open visualizations
# Look at evaluation_results/visualizations/

# Analyze in Python
python analyze_results.py evaluation_results/evaluation_results.json
```

## Summary

The evaluation script provides:
- ✅ IoU and Dice Coefficient metrics
- ✅ Precision and Recall scores
- ✅ Per-image detailed results
- ✅ Visual comparison images
- ✅ Statistical distribution analysis
- ✅ JSON export for further analysis
- ✅ COCO format support

Perfect for:
- Model validation
- Performance tracking
- Quality assurance
- Research and development
- Production monitoring

## Support

For issues:
1. Check this guide
2. Verify COCO JSON format
3. Inspect visualization images
4. Check model checkpoints
5. See README.md for general help

