"""Prepare WM811K data: create stratified splits and save indices."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import config
from src.data.dataset import WaferMapDataset
from src.data.preprocess import stratified_split, save_split_indices, compute_class_distribution
from src.utils.seed import seed_all


def parse_args() -> argparse.Namespace:
    """Parse CLI options for creating stratified splits."""
    parser = argparse.ArgumentParser(description="Prepare WM811K data")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw data root")
    parser.add_argument("--output", type=str, default="data/processed/split_indices.json", help="Where to store indices")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    paths = config.paths
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.raw_data
    dataset = WaferMapDataset(raw_dir)
    split = stratified_split(dataset.labels, seed=args.seed)
    save_split_indices(split, Path(args.output))

    distribution = compute_class_distribution(dataset.labels)
    print("Class distribution:", distribution)
    print("Saved splits to", args.output)


if __name__ == "__main__":
    main()
