"""Metrics for classification."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return average accuracy for a batch of logits and integer targets."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confusion matrix and return matrix with class indices."""
    cm = confusion_matrix(y_true, y_pred)
    classes = np.arange(cm.shape[0])
    return cm, classes


def compute_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return macro and weighted F1 scores."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
