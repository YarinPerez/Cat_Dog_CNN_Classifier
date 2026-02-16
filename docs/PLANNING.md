# Planning & Architecture Document

## System Architecture

```
Cat_Dog_CNN/
├── pyproject.toml              # uv project definition
├── .gitignore                  # Exclude data, models, caches
├── README.md                   # Results and usage instructions
├── CLAUDE.md                   # Project guidelines
├── docs/
│   ├── PRD.md                  # Product requirements
│   ├── TASKS.md                # Implementation tasks (checkboxes)
│   └── PLANNING.md             # This file
├── notebooks/
│   └── cat_dog_cnn.ipynb       # Main notebook (all code here)
├── data/                       # Dataset (downloaded at runtime)
│   └── PetImages/
│       ├── Cat/                # ~12,500 cat images
│       └── Dog/                # ~12,500 dog images
└── results/
    ├── plots/                  # PNG visualizations
    ├── metrics/                # JSON/TXT metric files
    └── models/                 # Model checkpoint (.pth)
```

## CNN Architecture Design

### Convolutional Blocks (x5)
Each block doubles channel count and halves spatial dimensions:

```
Input(3, 224, 224)
  -> Block1: 3 -> 32 channels, output 112x112
  -> Block2: 32 -> 64 channels, output 56x56
  -> Block3: 64 -> 128 channels, output 28x28
  -> Block4: 128 -> 256 channels, output 14x14
  -> Block5: 256 -> 512 channels, output 7x7
```

**Block pattern:** Conv3x3(pad=1) -> BN -> ReLU -> Conv3x3(pad=1) -> BN -> ReLU -> MaxPool2x2 -> Dropout2d(0.25)

### Classification Head
```
AdaptiveAvgPool2d(1)  -> (batch, 512, 1, 1)
Flatten               -> (batch, 512)
Linear(512, 256)      -> ReLU -> Dropout(0.5)
Linear(256, 2)        -> logits for [cat, dog]
```

### Parameter Count
- Conv layers: ~4.5M params
- FC layers: ~0.2M params
- Total: ~4.7M parameters

## Data Pipeline Design

### Transform Strategy
**Problem:** `ImageFolder` applies a single transform, but we need different transforms for train vs. eval splits.

**Solution:** `TransformSubset` wrapper class:
```python
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label
    def __len__(self):
        return len(self.subset)
```

Load `ImageFolder` with `transform=None`, use `random_split` for 70/15/15, then wrap each split with appropriate transforms.

### Augmentations (Training Only)
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- ToTensor + ImageNet Normalize

### Eval Transform
- Resize(256) -> CenterCrop(224)
- ToTensor + ImageNet Normalize

## Training Strategy

### Optimization
- **AdamW** with lr=1e-3, weight_decay=1e-4
- **CosineAnnealingLR** with T_max=epochs for smooth LR decay
- **GradScaler** for mixed precision (float16 forward pass, float32 gradients)
- **torch.compile()** for kernel fusion (satisfies CLAUDE.md compilation requirement)

### Early Stopping
- Monitor validation loss
- Patience = 10 epochs
- Save best model checkpoint when val loss improves
- Restore best model for final test evaluation

### Resource Management
- Train batch size: 128 (with AMP, fits comfortably in 24GB)
- Eval batch size: 256 (no gradient storage needed)
- DataLoader workers: 4
- pin_memory: True for faster GPU transfers

## Notebook Cell Structure

The notebook uses focused, short cells (<30 lines of code each) with markdown cells providing educational context. Total: 19 cells alternating between explanation and implementation.

## Task Tracking

Tasks are tracked in `docs/TASKS.md` with checkboxes. Each checkbox is marked `[x]` upon completion during implementation.
