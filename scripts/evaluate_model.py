"""Evaluate a saved checkpoint on the test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import config
from src.data.dataset import WaferMapDataset, WaferTransforms
from src.data.preprocess import load_split_indices
from src.models import MLPClassifier, SmallCNN, DeepCNN
from src.training.metrics import compute_accuracy, compute_confusion, compute_f1_scores
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_all


MODEL_REGISTRY = {
    "mlp": MLPClassifier,
    "cnn_small": SmallCNN,
    "cnn_deep": DeepCNN,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate WM811K model")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=config.training.image_size)
    parser.add_argument("--num-channels", type=int, default=config.training.num_channels)
    parser.add_argument("--split-file", type=str, default="data/processed/split_indices.json")
    parser.add_argument("--device", type=str, default=config.training.device)
    parser.add_argument("--seed", type=int, default=config.training.seed)
    return parser.parse_args()


def build_model(name: str, num_channels: int, num_classes: int, image_size: int) -> torch.nn.Module:
    if name == "mlp":
        return MLPClassifier((num_channels, image_size, image_size), num_classes)
    if name == "cnn_small":
        return SmallCNN(num_channels, num_classes)
    if name == "cnn_deep":
        return DeepCNN(num_channels, num_classes)
    raise ValueError(f"Unknown model {name}")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    logger = get_logger("evaluate")

    paths = config.paths
    transforms = WaferTransforms(args.image_size, args.num_channels, augment=False)
    dataset = WaferMapDataset(paths.raw_data, transform=transforms.eval())
    split = load_split_indices(Path(args.split_file))
    test_ds = WaferMapDataset(paths.raw_data, indices=split.test_indices, transform=transforms.eval(), label_map=dataset.label_map)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    model = build_model(args.model, args.num_channels, config.training.num_classes, args.image_size)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    y_true, y_pred = [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(test_loader)
    y_true_arr = torch.tensor(y_true)
    y_pred_arr = torch.tensor(y_pred)
    accuracy = (y_true_arr == y_pred_arr).float().mean().item()
    cm, classes = compute_confusion(y_true_arr.numpy(), y_pred_arr.numpy())
    f1_scores = compute_f1_scores(y_true_arr.numpy(), y_pred_arr.numpy())

    logger.info("Test loss %.4f accuracy %.4f", avg_loss, accuracy)
    logger.info("F1 scores: %s", f1_scores)
    print({"loss": avg_loss, "accuracy": accuracy, **f1_scores})


if __name__ == "__main__":
    main()
