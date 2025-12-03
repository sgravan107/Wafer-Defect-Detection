"""Training loop for wafer map classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.training.metrics import compute_accuracy, compute_confusion, compute_f1_scores
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_confusion_matrix, plot_training_curves, save_history


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
) -> Tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += compute_accuracy(outputs, targets)
        if batch_idx % log_interval == 0:
            tqdm.write(f"Batch {batch_idx}: loss {loss.item():.4f}")

    return running_loss / len(dataloader), running_acc / len(dataloader)


def evaluate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate the model for one epoch without gradient updates."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_acc += compute_accuracy(outputs, targets)
    return running_loss / len(dataloader), running_acc / len(dataloader)


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    log_interval: int,
    history_path: Path,
    checkpoint_path: Path,
    class_names: Tuple[str, ...] | None = None,
) -> Dict[str, float]:
    """Train a model, save the best checkpoint, and return test metrics."""
    logger = get_logger("train")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, log_interval)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info("Train loss %.4f acc %.4f | Val loss %.4f acc %.4f", train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Saved new best model to %s", checkpoint_path)

    save_history(history, history_path)
    plot_training_curves(history_path)

    # Load best model for test evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())

    y_true_arr = torch.tensor(y_true).numpy()
    y_pred_arr = torch.tensor(y_pred).numpy()
    cm, classes = compute_confusion(y_true_arr, y_pred_arr)
    f1_scores = compute_f1_scores(y_true_arr, y_pred_arr)

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        **f1_scores,
    }
    with history_path.with_suffix(".metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    class_labels = class_names if class_names is not None else tuple(str(c) for c in classes)
    plot_confusion_matrix(cm, list(class_labels), history_path.with_suffix("_cm.png"))
    logger.info("Test loss %.4f acc %.4f", test_loss, test_acc)
    return metrics
