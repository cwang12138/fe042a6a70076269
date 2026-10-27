"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2025/6/12 22:56
# @Author  : Chen Wang
# @Site    : 
# @File    : PeriodPatternMemorizer.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from einops import rearrange
from utils.metrics import residual_interaction_loss
from utils.tools import memory_report


class PeriodPatternMemorizer(nn.Module):
    def __init__(self, num_pattern, num_sensors, period_mem_dim, day_slots, init_values):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_pattern = num_pattern
        self.period_mem_dim = period_mem_dim
        self.day_slots = day_slots

        self.temp_pattern_memory = nn.Parameter(
            torch.tensor(init_values, dtype=torch.float32).unsqueeze(-1),
            requires_grad=True
        )       # (num_pattern, day_slots, num_sensors, period_mem_dim)
        self.register_buffer('sensor_idx', torch.arange(num_sensors, dtype=torch.long))
        self.register_buffer('pattern_memory', torch.zeros_like(self.temp_pattern_memory, dtype=torch.float32))
        self.is_fixed = False

    def fix_pattern_memory(self, scaler=None):
        self.is_fixed = True
        self.pattern_memory = self.temp_pattern_memory.data.clone().detach()
        if scaler is not None:
            self.pattern_memory = scaler.inverse_transform(self.pattern_memory)
        self.temp_pattern_memory.requires_grad = False

    def set_retrieval_scaler(self, scaler):
        # The aligned experiment stores raw prototypes and normalizes retrieval for Stage 2.
        assert self.is_fixed, "Fix raw pattern memory before setting its retrieval scaler."
        self.register_buffer('retrieval_memory', scaler.transform(self.pattern_memory).detach())

    def debug(self, save_path):
        """
        get general pattern memory info for debug purposes
        :return: numpy.ndarray, (num_pattern, day_slots, num_sensors, period_mem_dim), here, period_mem_dim is 1.
        """
        np.save(f"{save_path}/pattern_memory.npy", self.pattern_memory.cpu().detach().numpy())
        if hasattr(self, 'retrieval_memory'):
            np.save(f"{save_path}/retrieval_memory.npy", self.retrieval_memory.cpu().detach().numpy())

    def get_pattern_features(self, time_info):
        """
        Get the general pattern. (After self-supervised learning)
        :param time_info: torch.Tensor, (B, L, 2), [pattern_idx, slot_idx]
        :return: general pattern output, torch.Tensor, (B, L, S, period_mem_dim)
        """
        pattern_idx = time_info[..., 0].long()                # (B, L)
        slot_idx = time_info[..., 1].long()                   # (B, L)

        pattern_memory = getattr(self, 'retrieval_memory', self.pattern_memory)
        memory = pattern_memory[
            pattern_idx.unsqueeze(-1),                          # (B, L, 1)
            slot_idx.unsqueeze(-1),                             # (B, L, 1)
            self.sensor_idx.reshape(1, 1, -1)                   # (1, 1, S)
        ]
        return memory

    def forward(self, x1, x2, pattern_ids):
        """
        Update general pattern via self-supervised learning.
        :param x1: torch.Tensor, (B, day_slots, num_sensors)
        :param x2: torch.Tensor, (B, day_slots, num_sensors)
        :param pattern_ids: torch.LongTensor, (B, )
        :return: y1, y2, (B, day_slots, num_sensors)
        """
        assert self.is_fixed is False, "pattern memory is fixed, training is forbidden!"
        general_pattern = self.temp_pattern_memory[pattern_ids].squeeze(-1)
        y1 = x1 - general_pattern
        y2 = x2 - general_pattern
        return y1, y2


class SelfSupervisedDataLoader:
    def __init__(self, train_data, batch_size, day_slots, holiday_path, scaler, num_pattern=2,
                 pad_with_random_sample=True, shuffle=False, reshuffle_each_epoch=False):
        """

        :param train_data: pandas.Dataframe, shape (train_samples, num_sensors)
        """
        self.train_data = train_data
        self.holiday_path = holiday_path
        self.day_slots = day_slots
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.scaler = scaler
        self.num_pattern = num_pattern
        if self.num_pattern not in [2, 7]:
            raise ValueError(f"num_pattern={self.num_pattern} is not supported, only 2 or 7 are allowed.")

        self.holiday_dates = self._load_holidays()
        self.ssl_data, self.ssl_init_pattern, self.ssl_pattern_ids = self._preprocess_data()
        self.ssl_data = self.scaler.transform(self.ssl_data)
        self.ssl_init_pattern = self.scaler.transform(self.ssl_init_pattern)

        pair_indices_list = []
        for pattern_id in range(self.num_pattern):
            sample_indices = np.where(self.ssl_pattern_ids == pattern_id)[0]
            pairwise_indices = self._calculate_pairwise_indices(sample_indices)
            if pairwise_indices.shape[0] > 0:
                pair_indices_list.append(pairwise_indices)

        if len(pair_indices_list) > 0:
            self.indices = np.concatenate(pair_indices_list, axis=0)
        else:
            self.indices = np.empty((0, 2), dtype=np.int64)
        total_samples = self.indices.shape[0]
        self.original_pair_count = total_samples

        self.pad_with_random_sample = pad_with_random_sample
        if pad_with_random_sample:
            self.num_padding = (batch_size - (total_samples % batch_size)) % batch_size
            self.total_samples = total_samples + self.num_padding
        else:
            self.num_padding = 0
            self.total_samples = total_samples

        if self.num_padding > 0:
            rand_indices = np.random.choice(self.indices.shape[0], self.num_padding)
            self.indices = np.concatenate([self.indices, self.indices[rand_indices]], axis=0)

        if shuffle:
            perm = np.random.permutation(len(self.indices))
            self.indices = self.indices[perm]

        self.num_batch = (self.total_samples + self.batch_size - 1) // self.batch_size

    def get_iterator(self):
        self.current_ind = 0
        if self.shuffle and self.reshuffle_each_epoch:
            self.indices = self.indices[np.random.permutation(len(self.indices))]

        while self.current_ind < self.num_batch:
            start = self.current_ind * self.batch_size
            end = min((self.current_ind + 1) * self.batch_size, self.total_samples)
            batch_indices = self.indices[start: end]    # (batch, 2)

            x1_batch = self.ssl_data[batch_indices[:, 0]]
            x2_batch = self.ssl_data[batch_indices[:, 1]]
            left_pattern_ids = self.ssl_pattern_ids[batch_indices[:, 0]]
            right_pattern_ids = self.ssl_pattern_ids[batch_indices[:, 1]]
            if not np.array_equal(left_pattern_ids, right_pattern_ids):
                raise ValueError("Pair labels are inconsistent, please check pair construction logic.")
            pattern_id_batch = left_pattern_ids

            yield (x1_batch, x2_batch, pattern_id_batch)
            self.current_ind += 1

    def debug(self, save_path):
        """
        get several vital params for debug purpose
        :param save_path:
        :return:
        ssl_init_pattern: initial general pattern, numpy.ndarray, (num_pattern, day_slots, num_sensors)
        ssl_data: all the original training data, numpy.ndarray, (num_ssl_samples, day_slots, num_sensors)
        ssl_pattern_ids: pattern ids corresponding to ssl_data samples, numpy.ndarray, shape (num_ssl_samples, )
        """
        np.save(f"{save_path}/ssl_init_pattern.npy", self.ssl_init_pattern)
        np.save(f"{save_path}/ssl_data.npy", self.ssl_data)
        np.save(f"{save_path}/ssl_pattern_ids.npy", self.ssl_pattern_ids)

    def get_init_values(self):
        return self.ssl_init_pattern

    def _load_holidays(self):
        try:
            df = pd.read_csv(self.holiday_path, parse_dates=["date"])
            return set(df["date"].dt.date)
        except FileNotFoundError:
            print(f"Warning: Holiday file {self.holiday_path} not found")
            return set()

    def _calculate_pairwise_indices(self, indices):
        """
        C_n^2
        :param indices: numpy.ndarray, (num_indices,)
        :return: (num_indices_pairs, 2)
        """
        if len(indices) < 2:
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            i, j = np.triu_indices(len(indices), k=1)
            pairs = np.column_stack((indices[i], indices[j]))

        return pairs

    def _preprocess_data(self):
        """

        :return:
        filled_all_data, numpy.ndarray, shape (num_ssl_samples, day_slots, num_sensors);
        non_zero_mean (init general pattern), numpy.ndarray, shape (num_pattern, day_slots, num_sensors)
        valid_pattern_ids, numpy.ndarray, shape (num_ssl_samples,)
        """
        data = self.train_data.values       # (num_samples, num_sensors)
        data_dates = self.train_data.index  # (num_samples)

        # Filter incomplete calendar days before reshaping, so a time gap cannot shift later days.
        calendar_dates = data_dates.normalize()
        day_counts = calendar_dates.value_counts()
        complete_dates = day_counts.index[day_counts == self.day_slots]
        complete_rows = calendar_dates.isin(complete_dates)
        data = rearrange(data[complete_rows], '(n l) s -> n l s', l=self.day_slots)
        daily_timestamps = data_dates[complete_rows].to_numpy().astype('datetime64[ns]').reshape(-1, self.day_slots)
        data_dates = daily_timestamps[:, 0].astype('datetime64[D]')

        slot_duration = (pd.Timedelta(days=1) / self.day_slots).to_timedelta64()
        expected_timestamps = data_dates[:, None].astype('datetime64[ns]') + np.arange(self.day_slots) * slot_duration
        complete_days = np.all(daily_timestamps == expected_timestamps, axis=1)
        data = data[complete_days]
        data_dates = data_dates[complete_days]

        # pattern ids
        if self.num_pattern == 2:
            holiday_array = np.array([np.datetime64(d) for d in self.holiday_dates], dtype='datetime64[D]')
            pattern_ids = np.isin(data_dates, holiday_array).astype(np.int64)   # (0 - weekday, 1 - holiday)
        elif self.num_pattern == 7:
            # keep consistent with TimeInterpreter.get_time_info()
            pattern_ids = ((data_dates.view('int64') - 4) % 7).astype(np.int64)
        else:
            raise ValueError(f"num_pattern={self.num_pattern} is not supported")

        # process missing data
        missing_mask = (data == 0)
        missing_rate = np.mean(missing_mask, axis=(1, 2))       # (num_ssl_samples, )

        valid_mask = missing_rate <= 0.2
        valid_data = data[valid_mask]   # drop samples with beyond 20% missing rate
        valid_pattern_ids = pattern_ids[valid_mask]

        global_non_zero_mean = np.sum(valid_data, axis=0) / np.maximum((valid_data != 0).sum(axis=0), 1)
        non_zero_mean = np.zeros((self.num_pattern, self.day_slots, valid_data.shape[2]), dtype=valid_data.dtype)

        for pattern_id in range(self.num_pattern):
            pattern_indices = np.where(valid_pattern_ids == pattern_id)[0]
            if len(pattern_indices) == 0:
                non_zero_mean[pattern_id] = global_non_zero_mean
            else:
                pattern_data = valid_data[pattern_indices]
                non_zero_mean[pattern_id] = np.sum(pattern_data, axis=0) / np.maximum((pattern_data != 0).sum(axis=0), 1)

        filled_all_data = np.zeros_like(valid_data)
        for pattern_id in range(self.num_pattern):
            pattern_indices = np.where(valid_pattern_ids == pattern_id)[0]
            if len(pattern_indices) == 0:
                continue
            filled_all_data[pattern_indices] = np.where(
                valid_data[pattern_indices] == 0,
                non_zero_mean[pattern_id][None, :, :],
                valid_data[pattern_indices]
            )

        return filled_all_data, non_zero_mean, valid_pattern_ids


def train_ssl_general_pattern(dataloader, model, logger, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)
    total_time_cost = 0
    total_epoch = 100
    gpu_memory_used = {}

    logger.info("=== Before Training: Capturing General Pattern via Self-Supervised Learning ===")
    for epoch in range(total_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        model.train()
        data_iter = dataloader.get_iterator()
        losses = []
        for x1, x2, pattern_ids in data_iter:
            optimizer.zero_grad()
            x1 = torch.from_numpy(x1).float().to(device)
            x2 = torch.from_numpy(x2).float().to(device)
            pattern_ids = torch.from_numpy(pattern_ids).long().to(device)

            x1_residual, x2_residual = model(x1, x2, pattern_ids)
            loss = residual_interaction_loss(x1_residual, x2_residual)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        end_time = time.time()
        total_time_cost += (end_time - start_time)
        epoch_peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        gpu_memory_used[epoch + 1] = epoch_peak_mem
        if (epoch + 1) % 5 == 0:
            message = 'Epoch [{}/{}] train_loss: {:.4f}, lr: {:.6f}, {:.1f}s, {:.2f}GB'.format(
                epoch + 1, total_epoch, train_loss, optimizer.param_groups[0]['lr'], (end_time - start_time),
                epoch_peak_mem)
            logger.info(message)
    logger.info('The total training time is {:.4f} seconds!'.format(total_time_cost))
    memory_report(gpu_memory_used, logger)




