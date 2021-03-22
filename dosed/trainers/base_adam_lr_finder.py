""" Trainer class with Adam optimizer """

from torch import device
import torch.optim as optim
import copy
import tqdm
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets import collate
from ..functions import (loss_functions, available_score_functions, compute_metrics_dataset)
from ..utils import (match_events_localization_to_default_localizations, Logger)
import plotly.graph_objects as go

from .base_adam import TrainerBaseAdam


class LRFinderBaseAdam(TrainerBaseAdam):
    """ Trainer class with Adam optimizer """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_parameters = kwargs["optimizer_parameters"]

    def train(self, train_dataset, validation_dataset, batch_size=128):
        """ Metwork training with backprop """

        dataloader_parameters = {
            "num_workers": 0,
            "shuffle": True,
            "collate_fn": collate,
            "pin_memory": True,
            "batch_size": batch_size,
        }
        dataloader_train = DataLoader(train_dataset, **dataloader_parameters)
        losses = []
        lr_space = np.logspace(-6, -1, 200)
        t = tqdm.tqdm(lr_space)
        for lr, data in zip(t, cycle(dataloader_train)):
            t.set_postfix(current_lr=lr)
            self.optimizer_parameters.update({"lr": float(lr)})
            self.optimizer = optim.SGD(self.net.parameters(), **self.optimizer_parameters)
            # Set network to train mode
            self.net.train()

            (loss_classification_positive,
            loss_classification_negative,
            loss_localization) = self.get_batch_loss(data)

            loss = loss_classification_positive \
                + loss_classification_negative \
                + loss_localization
            loss.backward()

            # gradient descent
            self.optimizer.step()
            losses.append(loss.detach().cpu().item())

        fig = go.Figure(go.Scatter(x=lr_space, y=losses, mode="lines"))
        fig.update_layout(title="Learning rate finder", xaxis_title="learning rate", yaxis_title="loss")
        fig.update_xaxes(type="log")
        fig.write_html("lr_finder.html")
        return None, None, None