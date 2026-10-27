"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/8/23 7:28
# @Author  : Chen Wang
# @Site    : 
# @File    : utils.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import numpy as np


class DataLoader(object):
    def __init__(self, xs, ys, extra_inputs, batch_size, pad_with_last_sample=True, shuffle=False):
        """
        modify to reduce memory consumption from original version
        :param xs:
        :param ys:
        :param extra_inputs: dict-like extra inputs for each sample, for example:
         {
             'timestamps': ...,
             'x_missing_mask': ...,
             ...
         }
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.xs = xs
        self.ys = ys
        self.extra_inputs = extra_inputs
        assert isinstance(self.extra_inputs, dict), "extra_inputs must be a dict."
        self.original_size = len(xs)

        self.pad_with_last_sample = pad_with_last_sample
        if pad_with_last_sample:
            self.num_padding = (batch_size - (self.original_size % batch_size)) % batch_size
            self.total_samples = self.original_size + self.num_padding
        else:
            self.num_padding = 0
            self.total_samples = self.original_size

        self.indices = np.arange(self.original_size, dtype=np.uint32)
        if self.num_padding > 0:
            padding_indices = np.full(self.num_padding, self.original_size - 1, dtype=np.uint32)
            self.indices = np.concatenate([self.indices, padding_indices])
        if shuffle:
            np.random.shuffle(self.indices)

        self.num_batch = self.total_samples // self.batch_size

    def get_iterator(self):
        self.current_ind = 0

        while self.current_ind < self.num_batch:
            start = self.current_ind * self.batch_size
            end = (self.current_ind + 1) * self.batch_size
            batch_indices = self.indices[start:end]

            x_batch = self.xs[batch_indices]
            y_batch = self.ys[batch_indices]
            extra_inputs_batch = {
                key: value[batch_indices]
                for key, value in self.extra_inputs.items()
            }
            yield (x_batch, y_batch, extra_inputs_batch)
            self.current_ind += 1


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
