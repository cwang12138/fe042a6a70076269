"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 21:44
# @Author  : Chen Wang
# @Site    : 
# @File    : metrics.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import torch
import numpy as np

MASK_EPS = 0.0


def compute_roundtrip_zero_epsilon(mean, std):
    """
    Compute the exact pseudo-zero produced by the current float32
    standardize -> inverse_standardize round-trip for a raw zero.
    """
    mean32 = torch.tensor(float(mean), dtype=torch.float32)
    std32 = torch.tensor(float(std), dtype=torch.float32)
    if float(std32.item()) == 0.0:
        return 0.0
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    roundtrip = ((zero32 - mean32) / std32) * std32 + mean32
    return abs(float(roundtrip.item()))


def set_adaptive_mask_epsilon(mean, std):
    global MASK_EPS
    MASK_EPS = compute_roundtrip_zero_epsilon(mean, std)
    return MASK_EPS


def _resolve_mask_threshold(null_val=0.):
    if np.isnan(null_val):
        return null_val
    null_val = float(null_val)
    if abs(null_val) == 0.0:
        return MASK_EPS
    return abs(null_val)


def _build_np_mask(true, null_val=0., use_adaptive_mask=True):
    if np.isnan(null_val):
        mask = ~np.isnan(true)
    elif use_adaptive_mask:
        threshold = _resolve_mask_threshold(null_val)
        # Treat near-zero values as missing to avoid pseudo-zero round-trip noise
        # from standardize -> inverse_standardize corrupting the exact-zero mask.
        mask = np.abs(true) > threshold
    else:
        mask = np.not_equal(true, null_val)
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    return mask


def _build_torch_mask(y_true, null_val=0., use_adaptive_mask=True):
    if np.isnan(null_val):
        mask = ~torch.isnan(y_true)
    elif use_adaptive_mask:
        threshold = _resolve_mask_threshold(null_val)
        mask = torch.abs(y_true) > threshold
    else:
        mask = torch.ne(y_true, null_val)
    mask = mask.float()
    mask /= mask.mean()
    return mask


def residual_interaction_loss(I1, I2, lamda=1.0):
    """
    Pair-wise residual interaction loss for PPM calibration.

    Args:
        I1: torch.Tensor, shape (B, day_slots, num_sensors)
        I2: torch.Tensor, shape (B, day_slots, num_sensors)
        lamda: loss weight.

    Returns:
        torch.Tensor: scalar loss.
    """
    interaction = torch.abs((I1 * I2).sum(dim=1) / I1.shape[1])       # (B, num_sensors)
    return lamda * torch.mean(interaction)


# Followed the DCRNN numpy Implementation
def masked_rmse_np(pred, true, null_val=0., use_adaptive_mask=True):
    return np.sqrt(masked_mse_np(
        pred=pred, true=true, null_val=null_val, use_adaptive_mask=use_adaptive_mask
    ))


def masked_mse_np(pred, true, null_val=0., use_adaptive_mask=True):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_np_mask(true, null_val, use_adaptive_mask)
        mse = np.square(np.subtract(pred, true)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mape_np(pred, true, null_val=0., use_adaptive_mask=True):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_np_mask(true, null_val, use_adaptive_mask)
        error = np.subtract(pred, true).astype('float32')
        # Excluded zeros and round-trip pseudo-zeros must not enter the division.
        mape = np.zeros_like(error, dtype=np.result_type(error, true))
        np.divide(error, true, out=mape, where=mask > 0)
        mape = np.abs(mape)
        mape = np.nan_to_num(mask * mape)
        return 100 * np.mean(mape)


def masked_mae_np(pred, true, null_val=0., use_adaptive_mask=True):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_np_mask(true, null_val, use_adaptive_mask)
        mae = np.abs(np.subtract(pred, true)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def evaluate_all_metrics(pred, true, null_val=0., logger=None, use_adaptive_mask=True):
    """

    :param pred: (n_samples, n_sensors, horizon)
    :param true: (n_samples, n_sensors, horizon)
    :param null_val: np.nan or 0.0
    :param logger:
    :param use_adaptive_mask: whether to apply the STBoost near-zero threshold
    :return: three list(maes, rmses, mapes), len = n_horizon
    """
    horizon = true.shape[2]
    maes = []
    rmses = []
    mapes = []
    for idx in range(horizon):
        mae = masked_mae_np(pred[..., idx], true[..., idx], null_val, use_adaptive_mask)
        rmse = masked_rmse_np(pred[..., idx], true[..., idx], null_val, use_adaptive_mask)
        mape = masked_mape_np(pred[..., idx], true[..., idx], null_val, use_adaptive_mask)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        if logger is None:
            print('Horizon {}: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(idx + 1, mae, rmse, mape))
        else:
            logger.info('Horizon {}: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(idx + 1, mae, rmse, mape))
    if logger is None:
        print('Average Error: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(np.mean(maes),
                                                                                np.mean(rmses),
                                                                                np.mean(mapes)))
    else:
        logger.info('Average Error: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(np.mean(maes),
                                                                                      np.mean(rmses),
                                                                                      np.mean(mapes)))
    return maes, rmses, mapes


# DCRNN
def masked_mae_loss(y_pred, y_true, use_adaptive_mask=True):
    mask = _build_torch_mask(y_true, 0., use_adaptive_mask)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mse_loss(y_pred, y_true, use_adaptive_mask=True):
    mask = _build_torch_mask(y_true, 0., use_adaptive_mask)
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
