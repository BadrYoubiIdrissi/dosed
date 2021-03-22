import os
import torch
import wandb

import hydra
from omegaconf import OmegaConf

from dosed.utils import Compose, decode
from dosed.datasets import BalancedEventDataset as dataset
from dosed.models import DOSED3 as model
from dosed.datasets import get_train_validation_test
from dosed.trainers import trainers
from torchsummary import summary
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

from random import shuffle
import matplotlib

matplotlib.use("Agg")

@hydra.main(config_name="config")
def train(cfg):
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True), **cfg.wandb)
    
    train, validation, _ = get_train_validation_test(cfg.data.h5_directory,
                                                    percent_test=0,
                                                    percent_validation=cfg.data.percent_validation,
                                                    seed=cfg.job.seed)

    print("Number of records train:", len(train))
    print("Number of records validation:", len(validation))

    signals = [
        {
            'h5_path': signal,
            'fs': cfg.data.frequence_sample,
            'processing': {
                "type": "quantile_normalize",
                "args": {
                    "min_quantile": cfg.data.quantiles.min,
                    "max_quantile": cfg.data.quantiles.max
                }
            }
        } for signal in cfg.data.signals
    ]

    events = [
        {
            "name": event,
            "h5_path": event,
        }
        for event in cfg.data.events
    ]

    dataset_parameters = {
        "h5_directory": cfg.data.h5_directory,
        "signals": signals,
        "events": events,
        "window": cfg.data.window_size,
        "fs": cfg.data.frequence_sample,
        "ratio_positive": cfg.data.ratio_positive,
        "n_jobs": cfg.data.n_jobs,  # Make use of parallel computing to extract and normalize signals from h5
        "cache_data": cfg.data.cache,  # by default will store normalized signals extracted from h5 in h5_directory + "/.cache" directory
    }

    dataset_validation = dataset(records=validation, **dataset_parameters)
    transformations = hydra.utils.instantiate(cfg.data.transformations)
    dataset_parameters_train = {
        "transformations": Compose([trans for trans in transformations.values()])
    }
    dataset_parameters_train.update(dataset_parameters)
    dataset_train = dataset(records=train, **dataset_parameters_train)

    net_parameters = OmegaConf.to_container(cfg.model.init_params, resolve=True)
    
    net_parameters.update({
        "default_event_sizes": [
            default_event_size * cfg.data.frequence_sample
            for default_event_size in cfg.model.default_event_sizes
        ],
        "input_shape": dataset_train.input_shape,
        "number_of_classes": dataset_train.number_of_classes,
    })
    
    net = model(**net_parameters)
    net = net.to(cfg.job.device)
    print("Used model : ")
    print(summary(net, (len(cfg.data.signals),9000)))
    wandb.watch(net, log="all", log_freq=1)

    loss_specs = {
        "type": cfg.loss.type,
        "parameters": {
            "number_of_classes": dataset_train.number_of_classes,
            "device": cfg.job.device,
        }
    }

    save_callback = lambda epoch: net.save(cfg.trainer.save_folder + str(epoch) + "_net", net_parameters)
    def log_examples_1(epoch, name, dataset, sig_id=0):
        # nb_signals = 4
        fig = make_subplots(8, 8, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.001, vertical_spacing=0.001)
        fig.update_layout(showlegend=False, width=1000, height=1000)
        count = 0
        with torch.no_grad():
            for i in range(8):
                for j in range(8):
                    data_sample, labels = dataset[count]
                    count+=1
                    pred = net.predict(data_sample.unsqueeze(0).to(0))
                    fig.add_trace(go.Line(y=data_sample[sig_id]), row=j+1, col=i+1)

                    for true_event in labels:
                        fig.add_vrect(x0=true_event[0]*9000, x1=true_event[1]*9000, line_width=0, fillcolor="green", opacity=0.2, row=i+1, col=j+1)
                    if len(pred[0])>0:
                        for pred_event in pred[0]:
                            # import pdb; pdb.set_trace()
                            fig.add_vrect(x0=pred_event[0]*9000, x1=pred_event[1]*9000, line_width=0, fillcolor="red", opacity=0.2, row=i+1, col=j+1)

            fig.write_image(f"temp_{name}.png")
            wandb.log({"sample_predictions_"+name: wandb.Image(f"temp_{name}.png")}, commit=False)

    def log_examples(epoch, name, dataset, sig_id=0):
        # nb_signals = 4
        net.eval()
        fig, axs = plt.subplots(8,8, sharex=True, squeeze=True, figsize=(15,15))
        count = 0
        with torch.no_grad():
            for i in range(8):
                for j in range(8):
                    data_sample, labels = dataset[count]
                    count+=1
                    # if epoch == 2:
                    #     import pdb; pdb.set_trace()
                    pred = net.predict(data_sample.unsqueeze(0).to(0))
                    
                    axs[i,j].plot(data_sample[sig_id])
                    for true_event in labels:
                        axs[i,j].axvspan(true_event[0]*9000, true_event[1]*9000, facecolor="g", alpha=0.2)
                    if len(pred[0])>0:
                        for pred_event in pred[0]:
                            # import pdb; pdb.set_trace()
                            axs[i,j].axvspan(pred_event[0]*9000, pred_event[1]*9000, facecolor="r", alpha=0.2)

            wandb.log({"sample_predictions_"+name: wandb.Image(fig)}, commit=False)
            net.train()

    # def log_dataset(epoch, name, dataset):
    #     record = dataset_validation.records[1]
    #     predictions = net.predict_dataset(dataset)
    #     window_duration = 10
    #     sampling_frequency = 100
    #     for index_spindle in range(64):
    #         # retrive spindle at the right index
    #         spindle_start = float(predictions[record][0][index_spindle][0]) / sampling_frequency
    #         spindle_end = float(predictions[record][0][index_spindle][1]) / sampling_frequency

    #         # center data window on annotated spindle 
    #         start_window = spindle_start + (spindle_end - spindle_start) / 2 - window_duration
    #         stop_window = spindle_start + (spindle_end - spindle_start) / 2 + window_duration

    #         # Retrieve EEG data at right index
    #         index_start = int(start_window * sampling_frequency)
    #         index_stop = int(stop_window * sampling_frequency)
    #         y = dataset_validation.signals[record]["data"][0][index_start:index_stop]

    #         # Build corresponding time support
    #         t = start_window + np.cumsum(np.ones(index_stop - index_start) * 1 / sampling_frequency)

    def get_balanced_data(dataset, idx, number):
        data = []
        for i in range(number):
            data.append(dataset.extract_balanced_data(
                record=dataset.index_to_record_event[idx]["record"],
                max_index=dataset.index_to_record_event[idx]["max_index"],
                events_indexes=dataset.index_to_record_event[idx]["events_indexes"],
                no_events_indexes=dataset.index_to_record_event[idx]["no_events_indexes"]
            ))
        return data

    valid_sample = get_balanced_data(dataset_validation, 0, 64)
    train_sample = get_balanced_data(dataset_train, 0, 64)
    batch = next(dataset_train.get_record_batch(dataset_train.records[0], 500))[0].to(0)

    def output_hist(epoch, batch):
        out, _, _ = net(batch)
        wandb.log({"out_hist": out[:,0,0].detach().cpu()}, commit=False)

    shuffle(valid_sample)
    shuffle(train_sample)
    trainer = trainers[cfg.trainer.type](
        net=net,
        optimizer_parameters=cfg.optimizer_parameters,
        loss_specs=loss_specs,
        epochs=cfg.trainer.epochs,
        logger_parameters=cfg.logger,
        matching_overlap=cfg.trainer.matching_overlap,
        on_epoch_end_callbacks = [save_callback, 
                                  lambda epoch: log_examples(epoch, "train", train_sample), 
                                  lambda epoch: log_examples(epoch, "valid", valid_sample),
                                  lambda epoch: output_hist(epoch, batch)],
        loss_pos_weight=cfg.trainer.loss_pos_weight,
        loss_neg_weight=cfg.trainer.loss_neg_weight,
        loss_loc_weight=cfg.trainer.loss_loc_weight,
        lr_scheduler=cfg.lr_scheduler
    )
    best_net_train, best_metrics_train, best_threshold_train = trainer.train(
        dataset_train,
        dataset_validation,
        batch_size=cfg.trainer.batch_size
    )
    if best_net_train is not None:
        print("Best train metrics", best_metrics_train)
        print("Best threshold train", best_threshold_train)
        torch.save(best_net_train.state_dict(), "best_net.pt")

if __name__ == "__main__":
    train()