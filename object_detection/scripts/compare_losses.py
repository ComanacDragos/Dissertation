import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

experiments_paths = {
    "Reduced dataset": {
        "ClipNet": {
            "Baseline Loss": "outputs/kitti/fcos/loss/custom_arch_fixed_seed/original_loss/clipnorm=1/v1",
            "ClipLoss": "outputs/kitti/fcos/loss/custom_arch_fixed_seed/clip_loss/clipnorm=1/v1",
        },
        "Baseline Net": {
            "Baseline Loss": "outputs/kitti/fcos/loss/original_arch/original_loss",
            "ClipLoss": "outputs/kitti/fcos/loss/original_arch/clip_loss"
        }
    },
    "All labels": {
        "ClipNet": {
            "Baseline Loss": "outputs/kitti/fcos/loss/original_dataset/custom_arch/original_loss",
            "ClipLoss": "outputs/kitti/fcos/loss/original_dataset/custom_arch/clip_loss",

        },
        "BaselineNet": {
            "Baseline Loss": "outputs/kitti/fcos/loss/original_dataset/original_arch/original_loss",
            "ClipLoss": "outputs/kitti/fcos/loss/original_dataset/original_arch/clip_loss"
        }
    },
    "All labels without misc&don't care": {
        "ClipNet": {
            "Baseline Loss": "outputs/kitti/fcos/loss/original_dataset_without_misc_dont_care/custom_arch/original_loss",
            "ClipLoss": "outputs/kitti/fcos/loss/original_dataset_without_misc_dont_care/custom_arch/clip_loss",
        },
        "Baseline Net": {
            "Baseline Loss": "outputs/kitti/fcos/loss/original_dataset_without_misc_dont_care/original_arch/original_loss",
            "ClipLoss": "outputs/kitti/fcos/loss/original_dataset_without_misc_dont_care/original_arch/clip_loss"
        },
    },
}

ops = {
    'min': np.min,
    'max': np.max,
    'std': np.std,
    'mean': np.mean
}
rows = 2
cols = 2

root = "outputs/kitti/loss_analysis"

def reduce_data(csv, col, op):
    # reduced = {col: [] for col in set(csv.columns).difference({'lr', 'epoch', 'step'})}
    reduced = []
    epochs = csv.loc[:, 'epoch']
    # for col in reduced:
    values = csv.loc[:, col]
    current_epoch = 0
    epoch_values = []
    for value, epoch in zip(values, epochs):
        if epoch == current_epoch:
            epoch_values.append(value)
        else:
            reduced.append(op(epoch_values))
            current_epoch = epoch
            epoch_values = []
    return reduced


loss_types = ['loss', 'class_8', 'reg_8', 'centerness_8',
              'class_16', 'reg_16', 'centerness_16', 'class_32', 'reg_32',
              'centerness_32']
for dataset_type, experiments in experiments_paths.items():
    for network_type, name_to_paths in experiments.items():
        print(dataset_type, network_type)

        for loss_type in loss_types:
            plt.figure(figsize=(7, 7))
            plt.suptitle(f"{dataset_type} {network_type} {loss_type}")
            for name, path in name_to_paths.items():
                print('\t', name, path, loss_type)
                path = Path(path)
                train_data = pd.read_csv(path / "train" / "loss" / "train_per_step.csv")
                eval_data = pd.read_csv(path / "train" / "loss" / "eval_per_step.csv")

                for i, (op_name, op) in enumerate(ops.items(), start=1):
                    reduced_train = reduce_data(train_data, "train_loss" if loss_type == "loss" else loss_type, op)
                    reduced_eval = reduce_data(eval_data, "eval_loss" if loss_type == "loss" else loss_type, op)

                    plt.subplot(rows, cols, i)
                    plt.plot(reduced_train, label=f'{name} train')
                    plt.plot(reduced_eval, label=f'{name} eval')
                    plt.title(op_name)
                    plt.legend()

            plot_root = f"{root}/{dataset_type}/{network_type}"
            os.makedirs(plot_root, exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{plot_root}/{loss_type}.png")
            plt.clf()
            plt.close()
            # sys.exit(0)

