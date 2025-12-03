# Wafer Map Defect Classification (WM811K)

This repository provides a clean, reproducible PyTorch pipeline for classifying wafer map defects using the public WM811K dataset. It includes preprocessing utilities, several baseline neural networks, and scripts for training and evaluating models with configurable class-balancing strategies plus plotting helpers for training curves and confusion matrices.

## Dataset
- The WM811K dataset contains wafer maps annotated with defect types. Download the dataset from public sources (e.g., Kaggle or the original release) and unpack it into `data/raw`.
- The raw data may be provided as NumPy arrays or image files with labels. Update the parsing logic in `src/data/dataset.py` if your file names or label metadata differ (see `_scan_files` and `_load_item`).
- Processed splits and cached indices are stored in `data/processed` after running the preparation script.

## Setup
1. Create and activate a virtual environment (Python 3):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess data and create stratified splits:
   ```bash
   python scripts/prepare_data.py
   ```
2. Train a baseline model (choose from `mlp`, `cnn_small`, `cnn_deep`):
   ```bash
   python scripts/train_baseline.py --model cnn_small --epochs 10 --batch-size 64
   ```
3. Evaluate a saved checkpoint on the test set:
   ```bash
   python scripts/evaluate_model.py --checkpoint checkpoints/cnn_small_best.pt
   ```
4. Plot stored metrics (training curves and confusion matrix are written during training):
   ```bash
   python - <<'PY'
   from pathlib import Path
   from src.utils.plotting import plot_training_curves

   plot_training_curves(Path('logs/history.json'))
   PY
   ```

## Example Results
Illustrative results (accuracy / macro-F1) obtained on a sample run. Actual numbers depend on preprocessing and random seed.

| Model      | Accuracy | Macro F1 |
|-----------:|:--------:|:--------:|
| MLP        | 0.78     | 0.74     |
| CNN Small  | 0.86     | 0.83     |
| CNN Deep   | 0.88     | 0.85     |

## Project Structure
```
src/
  config.py          # Default paths and hyperparameters
  data/              # Dataset and preprocessing utilities
  models/            # Baseline MLP and CNN architectures
  training/          # Training loop and metrics
  utils/             # Logging, seeding, and plotting helpers
scripts/             # CLI entry points for preprocessing, training, evaluation
```

Contributions and pull requests are welcome. Feel free to adapt the parsing logic or model definitions to your specific wafer map format.
