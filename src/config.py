"""Default configuration for WM811K wafer map classification."""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Paths:
    """Filesystem paths used across the project."""
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data: Path = project_root / "data/raw"
    processed_data: Path = project_root / "data/processed"
    checkpoints: Path = project_root / "checkpoints"
    logs: Path = project_root / "logs"


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime options."""
    image_size: int = 64
    num_channels: int = 1
    num_classes: int = 9
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    weight_decay: float = 1e-4
    num_workers: int = 2
    balance_strategy: str = "sampler"  # choices: sampler, class_weights, none
    augment: bool = True
    seed: int = 42
    log_interval: int = 10
    device: str = "cuda"
    checkpoint_dir: Optional[Path] = None
    history_file: Optional[Path] = None
    class_weights: Optional[Tuple[float, ...]] = None

    def init_paths(self, paths: Paths) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = paths.checkpoints
        if self.history_file is None:
            self.history_file = paths.logs / "history.json"


@dataclass
class Config:
    """Root configuration object for convenience."""
    paths: Paths = field(default_factory=Paths)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = Config()
