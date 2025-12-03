"""Preprocessing helpers for WM811K wafer maps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class StratifiedSplit:
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


def compute_class_distribution(labels: List[int]) -> Dict[int, int]:
    distribution: Dict[int, int] = {}
    for label in labels:
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def stratified_split(
    labels: List[int], val_size: float = 0.1, test_size: float = 0.1, seed: int = 42
) -> StratifiedSplit:
    labels_array = np.array(labels)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(len(labels_array)), labels_array))
    labels_train_val = labels_array[train_val_idx]

    val_relative = val_size / (1 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(sss_val.split(np.zeros(len(labels_train_val)), labels_train_val))

    return StratifiedSplit(
        train_indices=train_val_idx[train_idx].tolist(),
        val_indices=train_val_idx[val_idx].tolist(),
        test_indices=test_idx.tolist(),
    )


def save_split_indices(split: StratifiedSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(split.__dict__, f, indent=2)


def load_split_indices(path: Path) -> StratifiedSplit:
    with path.open() as f:
        data = json.load(f)
    return StratifiedSplit(**data)


def balance_weights(labels: List[int]) -> List[float]:
    """Return inverse-frequency class weights for use in WeightedRandomSampler or loss."""
    counts = compute_class_distribution(labels)
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[label]) for label in labels]
    return weights


def indices_to_frame(indices: StratifiedSplit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_indices": indices.train_indices,
            "val_indices": indices.val_indices,
            "test_indices": indices.test_indices,
        }
    )
