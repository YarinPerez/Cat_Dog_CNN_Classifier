# Product Requirements Document: Cat vs Dog CNN Classifier

## 1. Project Overview

This project builds a **binary image classifier** that distinguishes between cats and dogs using a custom Convolutional Neural Network (CNN) implemented in PyTorch. The entire workflow lives in a Jupyter notebook for interactive exploration and learning.

### Learning Goals
- Understand how CNNs process images through convolutional, pooling, and fully-connected layers
- Learn practical data pipeline construction (augmentation, normalization, train/val/test splits)
- Experience modern PyTorch training techniques: mixed precision, torch.compile, learning rate scheduling
- Interpret model performance through confusion matrices, per-class metrics, and prediction visualization

## 2. Dataset

**Microsoft Cats vs Dogs Dataset**
- ~25,000 labeled images (12,500 cats + 12,500 dogs)
- JPEG format, variable resolutions
- Known issue: ~1,738 corrupt/truncated images that must be filtered before training
- Source: Microsoft Research, publicly available via direct download

### Data Splits
| Split      | Proportion | Purpose                        |
|------------|-----------|--------------------------------|
| Training   | 70%       | Model weight optimization       |
| Validation | 15%       | Hyperparameter tuning, early stopping |
| Test       | 15%       | Final unbiased performance evaluation |

## 3. Model Architecture

**Custom 4-Block CNN (~1.1M parameters)**

Each convolutional block follows the pattern:
```
Conv2d(3x3) -> BatchNorm -> ReLU -> Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.25)
```

| Block | Input Channels | Output Channels | Output Size (224 input) |
|-------|---------------|----------------|------------------------|
| 1     | 3             | 32             | 112x112               |
| 2     | 32            | 64             | 56x56                 |
| 3     | 64            | 128            | 28x28                 |
| 4     | 128           | 256            | 14x14                 |

**Classification Head:**
```
AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 2)
```

### Why This Architecture?
- **4 blocks** provide enough depth to learn hierarchical features (edges -> textures -> parts -> objects)
- **Batch normalization** stabilizes training and allows higher learning rates
- **Dropout** prevents overfitting on a relatively small dataset
- **Global average pooling** reduces parameters vs. flattening and adds spatial invariance
- **~1.1M params** is small enough to train quickly yet large enough for good accuracy

## 4. Training Configuration

| Parameter         | Value          | Rationale                                      |
|-------------------|---------------|-------------------------------------------------|
| Optimizer         | AdamW          | Weight decay decoupled from gradient updates    |
| Learning Rate     | 1e-3           | Standard starting point for Adam-family          |
| Weight Decay      | 1e-4           | Mild regularization                              |
| Scheduler         | CosineAnnealingLR | Smooth decay, good final convergence          |
| Epochs            | 30             | Sufficient with early stopping                   |
| Batch Size (train)| 128            | Conservative for 24GB VRAM with mixed precision  |
| Batch Size (eval) | 256            | No gradients stored, can use larger batches      |
| Early Stopping    | Patience=7     | Stop if val loss doesn't improve for 7 epochs    |
| Mixed Precision   | torch.amp      | ~2x speedup on RTX 4090, lower VRAM usage       |
| Compilation       | torch.compile  | Kernel fusion for additional speedup             |
| Image Size        | 224x224        | Standard size, compatible with transfer learning |

## 5. Deliverables

### Primary Output
- `notebooks/cat_dog_cnn.ipynb` - Complete annotated Jupyter notebook

### Saved Artifacts
- `results/models/best_model.pth` - Best model checkpoint (by validation loss)
- `results/plots/training_curves.png` - Loss and accuracy over epochs
- `results/plots/confusion_matrix.png` - Test set confusion matrix heatmap
- `results/plots/sample_predictions.png` - 4x4 grid of predictions with confidence
- `results/plots/per_class_metrics.png` - Bar chart of precision/recall/F1 per class
- `results/plots/lr_schedule.png` - Learning rate over epochs
- `results/metrics/classification_report.txt` - Full sklearn classification report
- `results/metrics/training_history.json` - Per-epoch train/val loss and accuracy
- `results/metrics/final_metrics.json` - Final test metrics summary

## 6. Success Criteria

| Metric          | Target   |
|-----------------|----------|
| Test Accuracy   | >= 90%   |
| Test F1-Score   | >= 0.90  |
| All plots saved | Yes      |
| All metrics saved | Yes    |
| Notebook runs end-to-end | Yes |

## 7. Visualization Requirements

### Confusion Matrix
- Seaborn heatmap with annotations
- Normalized and raw counts
- Clear class labels (Cat, Dog)

### Training Curves
- Dual-axis or subplot: loss curves (train/val) + accuracy curves (train/val)
- Marked best epoch
- Learning rate schedule in separate subplot

### Sample Predictions
- 4x4 grid showing test images
- Green border for correct, red for incorrect predictions
- Show predicted class and confidence percentage

### Per-Class Metrics
- Grouped bar chart showing precision, recall, and F1 for each class
- Numerical values displayed on bars

## 8. Hardware Requirements

- NVIDIA GPU with CUDA support (developed on RTX 4090, 24GB VRAM)
- ~5GB disk space for dataset
- Python 3.10+ with uv package manager
