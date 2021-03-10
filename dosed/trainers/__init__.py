from .base import TrainerBase
from .base_adam import TrainerBaseAdam
from .base_adam_lr_finder import LRFinderBaseAdam

__all__ = [
    "TrainerBase",
    "TrainerBaseAdam",
]

trainers = {
    "basic": TrainerBase,
    "adam": TrainerBaseAdam,
    "lr_finder": LRFinderBaseAdam
}
