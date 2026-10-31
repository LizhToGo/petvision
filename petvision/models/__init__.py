"""Neural network definitions for classification and segmentation."""

from .classifier import SmallCNN
from .unet import UNet

__all__ = ["SmallCNN", "UNet"]
