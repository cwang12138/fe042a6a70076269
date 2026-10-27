"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/10 18:24
# @Author  : Chen Wang
# @Site    : 
# @File    : tools.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import random

import numpy as np
import torch


def convert_standardized_scale(data, source_scaler, target_scaler):
    """Bridge standardized coordinates without a raw-scale intermediate tensor."""
    source_mean, source_std = float(source_scaler.mean), float(source_scaler.std)
    target_mean, target_std = float(target_scaler.mean), float(target_scaler.std)
    if source_mean == target_mean and source_std == target_std:
        return data
    scale = source_std / target_std
    offset = (source_mean - target_mean) / target_std
    return data * scale + offset


def save_rng_state(device):
    state = {
        "python_state": random.getstate(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_state": np.random.get_state()
    }
    return state


def restore_rng_state(state, device):
    random.setstate(state["python_state"])
    torch.set_rng_state(state["cpu_rng_state"])

    if torch.cuda.is_available():
        if state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)

        if state["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    np.random.set_state(state["numpy_state"])


def memory_report(memory_dict, logger):
    """
    Report memory usage (PyTorch torch.cuda.max_memory_allocated())
    :param memory_dict: Dictionary of epoch-wise peak GPU memory usage (unit: GB)
    :param logger:
    :return:
    """
    all_values = list(memory_dict.values())
    filtered_values = [v for k, v in memory_dict.items() if k != 1]

    # With first epoch
    range_with_first = (min(all_values), max(all_values))
    avg_with_first = sum(all_values) / len(all_values)

    # Without first epoch
    range_without_first = (min(filtered_values), max(filtered_values)) if filtered_values else (0, 0)
    avg_without_first = sum(filtered_values) / len(filtered_values) if filtered_values else 0

    # Get epoch range strings
    epoch_range_all = f"epochs 1-{len(all_values)}"
    epoch_range_filtered = f"epochs 2-{len(all_values)}" if len(all_values) > 1 else "epoch 2 (no data)"

    # Log formatted outputs
    logger.info(
        f"[Memory Report] Peak GPU memory usage range ({epoch_range_all}): [min: {range_with_first[0]:.2f} GB, max: {range_with_first[1]:.2f} GB]")
    logger.info(
        f"[Memory Report] Peak GPU memory usage range ({epoch_range_filtered}): [min: {range_without_first[0]:.2f} GB, max: {range_without_first[1]:.2f} GB]")
    logger.info(f"[Memory Report] Average peak GPU memory usage ({epoch_range_all}): {avg_with_first:.2f} GB")
    logger.info(f"[Memory Report] Average peak GPU memory usage ({epoch_range_filtered}): {avg_without_first:.2f} GB")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
