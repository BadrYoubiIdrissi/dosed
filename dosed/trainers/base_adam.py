""" Trainer class with Adam optimizer """

from torch import device
import torch.optim as optim

from .base import TrainerBase


class TrainerBaseAdam(TrainerBase):
    """ Trainer class with Adam optimizer """

    def __init__(self, *args, **kwargs):
        super(TrainerBaseAdam, self).__init__(*args, **kwargs)
        self.optimizer = optim.Adam(kwargs["net"].parameters(), **kwargs["optimizer_parameters"])
