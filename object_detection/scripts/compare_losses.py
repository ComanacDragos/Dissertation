import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

for dataset_type, experiments in experiments_paths.items():
    for network_type, name_to_paths in experiments.items():
        print(dataset_type, network_type)

        plt.title(f"{dataset_type} {network_type}")
        for name, path in name_to_paths.items():
            print('\t', name, path)
            path = Path(path)
            train_data = pd.read_csv(path / "train" / "loss" / "train_per_step.csv")
            eval_data = pd.read_csv(path / "train" / "loss" / "eval_per_step.csv")



