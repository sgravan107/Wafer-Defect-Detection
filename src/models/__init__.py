"""Model zoo for wafer map classification."""
from .mlp import MLPClassifier
from .cnn_small import SmallCNN
from .cnn_deep import DeepCNN

__all__ = ["MLPClassifier", "SmallCNN", "DeepCNN"]
