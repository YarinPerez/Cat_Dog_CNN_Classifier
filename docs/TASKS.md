# Implementation Tasks

## Project Setup
- [x] Create `pyproject.toml` with all dependencies
- [x] Create `.gitignore`
- [x] Create directory structure (`notebooks/`, `data/`, `results/plots/`, `results/metrics/`, `results/models/`)
- [x] Run `uv sync` to install dependencies

## Jupyter Notebook Implementation
- [x] Cell 1: Title & Introduction (markdown)
- [x] Cell 2: Imports & Configuration
- [x] Cell 3: GPU Check
- [x] Cell 4: Dataset Download & Extraction
- [x] Cell 5: Corrupt Image Filtering
- [x] Cell 6: Dataset Exploration (class counts, sample grid)
- [x] Cell 7: Data Transforms (markdown explanation)
- [x] Cell 8: Dataset & DataLoader Setup (ImageFolder, splits, TransformSubset, loaders)
- [x] Cell 9: CNN Architecture Definition + torchinfo summary
- [x] Cell 10: Training Setup (loss, optimizer, scheduler, scaler, compile)
- [x] Cell 11: Training Loop (train/val per epoch, early stopping, checkpointing)
- [x] Cell 12: Training Curves (plot & save loss/accuracy/LR)
- [x] Cell 13: Load Best Model & Test Evaluation
- [x] Cell 14: Classification Report (sklearn, save to TXT)
- [x] Cell 15: Confusion Matrix (seaborn heatmap, save PNG)
- [x] Cell 16: Sample Predictions (4x4 grid with color borders)
- [x] Cell 17: Per-Class Metrics (bar chart, save PNG)
- [x] Cell 18: Summary & Conclusions (markdown)
- [x] Cell 19: Save All Results (training_history.json, final_metrics.json)

## Finalization
- [x] Run notebook end-to-end
- [x] Verify all output files exist
- [x] Write README.md with results and embedded plots
- [x] Mark all tasks as done
