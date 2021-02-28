import os
import json

import torch
import tempfile
import json
import random

import hydra

from dosed.utils import Compose
from dosed.datasets import BalancedEventDataset as dataset
from dosed.models import DOSED3 as model
from dosed.datasets import get_train_validation_test
from dosed.trainers import trainers
from dosed.preprocessing import GaussianNoise, RescaleNormal, Invert
from torchsummary import summary

@hydra.main(config_name="config")
def train(cfg):
    seed = 2019
    h5_directory = '/gpfs/users/idrissib/datasets/sleepapnea/records/train'  # adapt if you used a different DOWNLOAD_PATH when running `make download_example`

    train, validation, _ = get_train_validation_test(h5_directory,
                                                    percent_test=0,
                                                    percent_validation=33,
                                                    seed=seed)

    print("Number of records train:", len(train))
    print("Number of records validation:", len(validation))

    window = 90  # window duration in seconds
    ratio_positive = 0.5  # When creating the batch, sample containing at least one spindle will be drawn with that probability

    fs = 100
    signals = ['abdom_belt','airflow','PPG','thorac_belt','snore','SPO2','C4-A1','O2-A1']
    ranges = [(-1000.0, 1000.0), (-400.0, 400.0), (-4000.0, 4000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (0.0, 100000.0), (-1500.0, 1500.0), (-1500.0, 1500.0)]

    signals = [
        {
            'h5_path': signal,
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": min_value,
                    "max_value": max_value,
                }
            }
        } for signal, (min_value, max_value) in zip(signals, ranges)
    ]

    events = [
        {
            "name": "apnea",
            "h5_path": "apnea",
        }
    ]

    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": window,
        "fs": fs,
        "ratio_positive": ratio_positive,
        "n_jobs": 16,  # Make use of parallel computing to extract and normalize signals from h5
        "cache_data": True,  # by default will store normalized signals extracted from h5 in h5_directory + "/.cache" directory
    }

    dataset_validation = dataset(records=validation, **dataset_parameters)
    # for training add data augmentation
    dataset_parameters_train = {
        "transformations": Compose([
            GaussianNoise(),
            RescaleNormal(),
            Invert(),
        ])
    }
    dataset_parameters_train.update(dataset_parameters)
    dataset_train = dataset(records=train, **dataset_parameters_train)

    default_event_sizes = [0.7, 1, 1.3]
    k_max = 5
    kernel_size = 5
    probability_dropout = 0.1
    device = torch.device("cuda")

    sampling_frequency = dataset_train.fs

    net_parameters = {
        "detection_parameters": {
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.3
        },
        "default_event_sizes": [
            default_event_size * sampling_frequency
            for default_event_size in default_event_sizes
        ],
        "k_max": k_max,
        "kernel_size": kernel_size,
        "pdrop": probability_dropout,
        "fs": sampling_frequency,   # just used to print architecture info with right time
        "input_shape": dataset_train.input_shape,
        "number_of_classes": dataset_train.number_of_classes,
    }
    net = model(**net_parameters)
    net = net.to(device)
    print("Used model : ")
    print(summary(net))

    optimizer_parameters = {
        "lr": 5e-3,
        "weight_decay": 1e-8,
    }
    loss_specs = {
        "type": "focal",
        "parameters": {
            "number_of_classes": dataset_train.number_of_classes,
            "device": device,
        }
    }
    epochs = 20

    trainer = trainers["adam"](
        net,
        optimizer_parameters=optimizer_parameters,
        loss_specs=loss_specs,
        epochs=epochs,
        logger_parameters={
                "num_events": 1,
                "output_dir": ".",
                "output_fname": 'train_history.json',
                "metrics": ["precision", "recall", "f1"],
                "name_events": ["apnea"]
            },
        matching_overlap=0.3,
        save_folder="./"
    )
    import pdb; pdb.set_trace()
    best_net_train, best_metrics_train, best_threshold_train = trainer.train(
        dataset_train,
        dataset_validation,
    )
    print("Best train metrics", best_metrics_train)
    print("Best threshold train", best_threshold_train)
    torch.save(best_net_train.state_dict(), "best_net.pt")

if __name__ == "__main__":
    train()