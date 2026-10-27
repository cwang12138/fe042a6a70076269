"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2025/4/18 17:00
# @Author  : Chen Wang
# @Site    : 
# @File    : TimeInterpreter.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import numpy as np
import pandas as pd
import torch


class TimeInterpreter:
    def __init__(self, path, p_len, q_len, device, duration, start_date):
        self.path = path
        self.p_len = p_len
        self.q_len = q_len
        self.device = device
        self.duration = duration
        self.start_date = start_date

        self.holiday_dates = self._load_holidays(path)
        self.timestamps = None
        self.time_encoding = None

    def _load_holidays(self, csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            return set(df["date"].dt.date)
        except FileNotFoundError:
            print(f"Warning: Holiday file {csv_path} not found")
            return set()

    def reset_timestamps(self):
        self.timestamps = None
        self.time_encoding = None

    def set_timestamps(self, start_timestamps):
        """
        :param start_timestamps: (num_samples,) or actual window timestamps (num_samples, P + Q)
        :param duration: interval between two timestamps, default to 5 minutes
        """
        if start_timestamps.ndim == 2:
            if start_timestamps.shape[1] != self.p_len + self.q_len:
                raise ValueError('Actual window timestamps must have P + Q columns.')
            self.timestamps = start_timestamps
        else:
            time_offsets = np.arange(self.p_len + self.q_len) * self.duration
            deltas = time_offsets.astype('timedelta64[m]')
            self.timestamps = start_timestamps[:, np.newaxis] + deltas
        self.time_encoding = self._calculate_time_encoding()

    def _calculate_time_encoding(self):
        """
        :return: encoding (num_samples, length, 3)
        """
        # Timestep: Convert to daily timestep index
        minutes_since_midnight = (self.timestamps - self.timestamps.astype('datetime64[D]')) / np.timedelta64(1, 'm')
        num_timesteps = int(24 * 60 / self.duration)                                          # e.g., 5 minutes → 288
        timestep_idx = (minutes_since_midnight / self.duration).astype(int) % num_timesteps   # [0, num_timesteps - 1]

        # Day of week: Monday=0, Sunday=6
        day_of_week_idx = (self.timestamps.astype('datetime64[D]').view('int64') - 4) % 7  # 1970-01-01 is Thursday

        dates = self.timestamps.astype('datetime64[D]')

        # Holiday
        holiday_array = np.array([np.datetime64(d) for d in self.holiday_dates], dtype='datetime64[D]')
        holiday_encoding = np.isin(dates, holiday_array).astype(np.int64)

        # Stack raw calendar indices: timestep, day of week, holiday
        encoding = np.stack([timestep_idx, day_of_week_idx, holiday_encoding], axis=-1)

        return encoding

    def get_time_encoding(self):
        """
        :return: time_encoding, torch, (num_samples, p + q, num_encoding(3))
        """
        return torch.tensor(self.time_encoding, dtype=torch.long, device=self.device, requires_grad=False)

    def get_timestamps(self):
        """
        :return: timestamps, torch, (num_samples, )
        """
        return self.timestamps

    def get_time_info(self, num_pattern):
        """

        :param num_pattern: the number of period temporal pattern to save
        :return: time_info shape (B, L, 2)
        """
        assert self.timestamps is not None, "timestamps must be initialized!"
        if num_pattern == 7:
            pattern_idx = (self.timestamps.astype('datetime64[D]').view('int64') - 4) % 7
        elif num_pattern == 2:
            pattern_idx = self.time_encoding[..., -1]
        else:
            raise ValueError(f"num_pattern={num_pattern} is not supported")
        minutes = (self.timestamps - self.timestamps.astype('datetime64[h]')) / np.timedelta64(1, 'm')
        hours = (self.timestamps.astype('datetime64[h]') - self.timestamps.astype(
            'datetime64[D]')) / np.timedelta64(1, 'h')
        slot_idx = (minutes + hours * 60) / self.duration
        return np.stack([pattern_idx, slot_idx], axis=-1)
