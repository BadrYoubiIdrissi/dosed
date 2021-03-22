""" Trainer class basic with SGD optimizer """

import copy
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets import collate
from ..functions import (loss_functions, available_score_functions, compute_metrics_dataset)
from ..utils import (match_events_localization_to_default_localizations, Logger)

from torch.autograd import grad
import wandb
class TrainerBase:
    """Trainer class basic """

    def __init__(
            self,
            net,
            optimizer_parameters={
                "lr": 0.001,
                "weight_decay": 1e-8,
            },
            loss_specs={
                "type": "focal",
                "parameters": {
                    "number_of_classes": 1,
                    "alpha": 0.25,
                    "gamma": 2,
                    "device": torch.device("cuda"),
                }
            },
            metrics=["precision", "recall", "f1"],
            epochs=100,
            metric_to_maximize="f1",
            patience=None,
            save_folder=None,
            logger_parameters={
                "num_events": 1,
                "output_dir": None,
                "output_fname": 'train_history.json',
                "metrics": ["precision", "recall", "f1"],
                "name_events": ["event_type_1"]
            },
            threshold_space={
                "upper_bound": 0.85,
                "lower_bound": 0.55,
                "num_samples": 5,
                "zoom_in": False,
            },
            matching_overlap=0.5,
            on_epoch_end_callbacks=[],
            loss_pos_weight=1.0,
            loss_neg_weight=1.0,
            loss_loc_weight=1.0,
            lr_scheduler=None
    ):

        self.net = net
        print("Device: ", net.device)
        self.loss_function = loss_functions[loss_specs["type"]](
            **loss_specs["parameters"])
        self.optimizer = optim.SGD(net.parameters(), **optimizer_parameters)
        self.metrics = {
            score: score_function for score, score_function in
            available_score_functions.items()
            if score in metrics + [metric_to_maximize]
        }
        self.epochs = epochs
        self.threshold_space = threshold_space
        self.metric_to_maximize = metric_to_maximize
        self.patience = patience if patience else epochs
        self.save_folder = save_folder
        self.matching_overlap = matching_overlap
        self.matching = match_events_localization_to_default_localizations
        self.on_epoch_end_callbacks = on_epoch_end_callbacks
        self.loss_pos_weight=loss_pos_weight
        self.loss_neg_weight=loss_neg_weight
        self.loss_loc_weight=loss_loc_weight
        if lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, **lr_scheduler)
        else:
            self.lr_scheduler = None
        if logger_parameters is not None:
            self.train_logger = Logger(**logger_parameters)

    def on_batch_start(self):
        pass

    def on_epoch_end(self, epoch):
        for callback in self.on_epoch_end_callbacks:
            callback(epoch)

    def validate(self, validation_dataset, threshold_space):
        """
        Compute metrics on validation_dataset net for test_dataset and
        select best classification threshold
        """
        best_thresh = -1
        best_metrics_epoch = {
            metric: -1
            for metric in self.metrics.keys()
        }

        # Compute predicted_events
        thresholds = np.sort(
            np.random.uniform(threshold_space["upper_bound"],
                              threshold_space["lower_bound"],
                              threshold_space["num_samples"]))

        for threshold in thresholds:
            metrics_thresh = compute_metrics_dataset(
                self.net,
                validation_dataset,
                threshold,
            )

            # If 0 events predicted, all superiors thresh's will also predict 0
            if metrics_thresh == -1:
                if best_thresh in (self.threshold_space["upper_bound"],
                                   self.threshold_space["lower_bound"]):
                    print(
                        "Best classification threshold is " +
                        "in the boundary ({})! ".format(best_thresh) +
                        "Consider extending threshold range")
                return best_metrics_epoch, best_thresh

            # Add to logger
            if "train_logger" in vars(self):
                self.train_logger.add_new_metrics((metrics_thresh, threshold))

            # Compute mean metric to maximize across events
            mean_metric_to_maximize = np.nanmean(
                [m[self.metric_to_maximize] for m in metrics_thresh])

            if mean_metric_to_maximize >= best_metrics_epoch[
                    self.metric_to_maximize]:
                best_metrics_epoch = {
                    metric: np.nanmean(
                        [m[metric] for m in metrics_thresh])
                    for metric in self.metrics.keys()
                }

                best_thresh = threshold

        if best_thresh in (threshold_space["upper_bound"],
                           threshold_space["lower_bound"]):
            print("Best classification threshold is " +
                  "in the boundary ({})! ".format(best_thresh) +
                  "Consider extending threshold range")

        return best_metrics_epoch, best_thresh

    def get_batch_loss(self, data):
        """ Single forward and backward pass """

        # Get signals and labels
        signals, events = data
        x = signals.to(self.net.device)
        x.requires_grad = True

        # Forward
        localizations, classifications, localizations_default = self.net.forward(x)

        wandb.log({"localiz_center": localizations[:,:,0].detach().cpu()}, commit=False)
        wandb.log({"localiz_dur": localizations[:,:,1].detach().cpu()}, commit=False)
        wandb.log({"classifications_pos": classifications[:,:,1].detach().cpu()}, commit=False)
        wandb.log({"classifications_neg": classifications[:,:,0].detach().cpu()}, commit=False)

        pos_mask = classifications.argmax(dim=-1)>0
        wandb.log({"localiz_center_pos": localizations[pos_mask][:,0].detach().cpu()}, commit=False)
        wandb.log({"localiz_dur_pos": localizations[pos_mask][:,1].detach().cpu()}, commit=False)


        # Matching
        localizations_target, classifications_target = self.matching(
            localizations_default=localizations_default,
            events=events,
            threshold_overlap=self.matching_overlap)
        localizations_target = localizations_target.to(self.net.device)
        classifications_target = classifications_target.to(self.net.device)

        # Loss
        (loss_classification_positive,
         loss_classification_negative,
         loss_localization) = (
             self.loss_function(localizations,
                                classifications,
                                localizations_target,
                                classifications_target))

        # import pdb; pdb.set_trace()
        # probe = lambda y, x: grad(y, x, retain_graph=True)[0].abs().mean().item()
        wandb.log({"input_sensitivity_loc": grad(loss_localization, x, retain_graph=True)[0].view(x.size(0), -1).norm(dim=-1).mean().item()}, commit=False)
        wandb.log({"input_sensitivity_neg": grad(loss_classification_negative, x, retain_graph=True)[0].view(x.size(0), -1).norm(dim=-1).mean().item()}, commit=False)
        wandb.log({"input_sensitivity_pos": grad(loss_classification_positive, x, retain_graph=True)[0].view(x.size(0), -1).norm(dim=-1).mean().item()}, commit=False)

        return loss_classification_positive, \
            loss_classification_negative, \
            loss_localization

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
        dataloader_val = DataLoader(validation_dataset, **dataloader_parameters)

        metrics_final = {
            metric: 0
            for metric in self.metrics.keys()
        }

        best_value = -np.inf
        best_threshold = None
        best_net = None
        counter_patience = 0
        last_update = None
        t = tqdm.tqdm(range(self.epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric_score=best_value,
                    threshold=best_threshold,
                    last_update=last_update,
                )

            epoch_loss_classification_positive_train = 0.0
            epoch_loss_classification_negative_train = 0.0
            epoch_loss_localization_train = 0.0

            epoch_loss_classification_positive_val = 0.0
            epoch_loss_classification_negative_val = 0.0
            epoch_loss_localization_val = 0.0

            for i, data in enumerate(dataloader_train, 0):
                # import pdb; pdb.set_trace()
                # On batch start
                self.on_batch_start()

                self.optimizer.zero_grad()

                # Set network to train mode
                self.net.train()

                (loss_classification_positive,
                 loss_classification_negative,
                 loss_localization) = self.get_batch_loss(data)

                epoch_loss_classification_positive_train += \
                    loss_classification_positive
                epoch_loss_classification_negative_train += \
                    loss_classification_negative
                epoch_loss_localization_train += loss_localization

                loss = self.loss_pos_weight*loss_classification_positive \
                    + self.loss_neg_weight*loss_classification_negative \
                    + self.loss_loc_weight*loss_localization
                loss.backward()

                # gradient descent
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
            
            epoch_loss_classification_positive_train /= (i + 1)
            epoch_loss_classification_negative_train /= (i + 1)
            epoch_loss_localization_train /= (i + 1)

            for i, data in enumerate(dataloader_val, 0):

                (loss_classification_positive,
                 loss_classification_negative,
                 loss_localization) = self.get_batch_loss(data)

                epoch_loss_classification_positive_val += \
                    loss_classification_positive
                epoch_loss_classification_negative_val += \
                    loss_classification_negative
                epoch_loss_localization_val += loss_localization

            epoch_loss_classification_positive_val /= (i + 1)
            epoch_loss_classification_negative_val /= (i + 1)
            epoch_loss_localization_val /= (i + 1)

            metrics_epoch, threshold = self.validate(
                validation_dataset=validation_dataset,
                threshold_space=self.threshold_space,
            )

            if self.threshold_space["zoom_in"] and threshold != -1:
                threshold_space_size = self.threshold_space["upper_bound"] - \
                    self.threshold_space["lower_bound"]
                zoom_metrics_epoch, zoom_threshold = self.validate(
                    validation_dataset=validation_dataset,
                    threshold_space={
                        "upper_bound": threshold + 0.1 * threshold_space_size,
                        "lower_bound": threshold - 0.1 * threshold_space_size,
                        "num_samples": self.threshold_space["num_samples"],
                    })
                if zoom_metrics_epoch[self.metric_to_maximize] > metrics_epoch[
                        self.metric_to_maximize]:
                    metrics_epoch = zoom_metrics_epoch
                    threshold = zoom_threshold

            if metrics_epoch[self.metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.metric_to_maximize]
                best_threshold = threshold
                last_update = epoch
                best_net = copy.deepcopy(self.net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics.keys()
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

            self.on_epoch_end(epoch)
            if "train_logger" in vars(self):
                self.train_logger.add_new_loss(
                    epoch_loss_localization_train.item(),
                    epoch_loss_classification_positive_train.item(),
                    epoch_loss_classification_negative_train.item(),
                    mode="train"
                )
                self.train_logger.add_new_loss(
                    epoch_loss_localization_val.item(),
                    epoch_loss_classification_positive_val.item(),
                    epoch_loss_classification_negative_val.item(),
                    mode="validation"
                )
                self.train_logger.add_current_metrics_to_history()
                self.train_logger.dump_train_history()

        return best_net, metrics_final, best_threshold
