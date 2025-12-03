"""Train baseline model for wafer map classification."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import config, TrainingConfig
from src.data.dataset import WaferMapDataset, WaferTransforms
from src.data.preprocess import load_split_indices, balance_weights
from src.models import MLPClassifier, SmallCNN, DeepCNN
from src.training.train import train_and_evaluate
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_all


MODEL_REGISTRY = {
    "mlp": MLPClassifier,
    "cnn_small": SmallCNN,
    "cnn_deep": DeepCNN,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train WM811K baseline")
    parser.add_argument("--model", type=str, default="cnn_small", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
    parser.add_argument("--lr", type=float, default=config.training.lr)
    parser.add_argument("--image-size", type=int, default=config.training.image_size)
    parser.add_argument("--num-channels", type=int, default=config.training.num_channels)
    parser.add_argument("--balance-strategy", type=str, default=config.training.balance_strategy, choices=["sampler", "class_weights", "none"])
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=config.training.augment)
    parser.add_argument("--split-file", type=str, default="data/processed/split_indices.json")
    parser.add_argument("--device", type=str, default=config.training.device)
    parser.add_argument("--seed", type=int, default=config.training.seed)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--history", type=str, default="logs/history.json")
    return parser.parse_args()


def build_model(name: str, num_channels: int, num_classes: int, image_size: int) -> nn.Module:
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
    logger = get_logger("train_baseline")

    paths = config.paths
    train_cfg = TrainingConfig(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_classes=config.training.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        balance_strategy=args.balance_strategy,
        augment=args.augment,
        seed=args.seed,
        device=args.device,
    )
    train_cfg.init_paths(paths)

    split = load_split_indices(Path(args.split_file))
    transforms = WaferTransforms(train_cfg.image_size, train_cfg.num_channels, augment=train_cfg.augment)
    dataset = WaferMapDataset(paths.raw_data, transform=transforms.train())
    
    train_ds = WaferMapDataset(paths.raw_data, indices=split.train_indices, transform=transforms.train(), label_map=dataset.label_map)
    val_ds = WaferMapDataset(paths.raw_data, indices=split.val_indices, transform=transforms.eval(), label_map=dataset.label_map)
    test_ds = WaferMapDataset(paths.raw_data, indices=split.test_indices, transform=transforms.eval(), label_map=dataset.label_map)

    class_names = tuple(sorted(dataset.label_map, key=lambda k: dataset.label_map[k]))

    sampler = None
    class_weights = None
    if train_cfg.balance_strategy == "sampler":
        weights = balance_weights([train_ds.labels[i] for i in range(len(train_ds))])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    elif train_cfg.balance_strategy == "class_weights":
        from collections import Counter

        counts = Counter(train_ds.labels)
        total = sum(counts.values())
        class_weights = torch.tensor([total / counts[i] for i in range(len(counts))], dtype=torch.float)
        class_weights = class_weights / class_weights.sum()

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    model = build_model(args.model, train_cfg.num_channels, train_cfg.num_classes, train_cfg.image_size).to(train_cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(train_cfg.device) if class_weights is not None else None)

    metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device(train_cfg.device),
        epochs=train_cfg.epochs,
        log_interval=train_cfg.log_interval,
        history_path=Path(args.history),
        checkpoint_path=Path(args.checkpoint),
        class_names=class_names,
    )
    logger.info("Finished training: %s", metrics)


if __name__ == "__main__":
    main()
