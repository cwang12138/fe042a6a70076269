"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 21:02
# @Author  : Chen Wang
# @Site    : 
# @File    : data_factory.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import numpy as np
import pandas as pd
import os


class Data_Traffic:
    def __init__(self, args, base_model, logger, is_enhanced):
        self.args = args
        self.base_model = base_model
        self.logger = logger
        self.is_enhanced = is_enhanced
        self.df = self._unified_load_dataset()
        self.active_categories = ['train', 'val', 'test']
        self.value_only = False


    def get_data(self, stage, categories=None, value_only=False):
        assert stage in [1, 2], f'stage must be 1 or 2, got {stage}'
        self.full_timestamps = stage == 2 and self.is_enhanced and bool(getattr(self.args, 'stb_stage2_align_inputs', 1))
        self.active_categories = self._normalize_categories(categories)
        self.value_only = value_only
        data = self._preprocess_data(stage)
        self.active_categories = ['train', 'val', 'test']
        self.value_only = False
        return data


    def get_train_dataset(self):
        return self.df['train']


    def get_boost_adj(self):
        return self._process_STBoost_adj()


    @staticmethod
    def _normalize_categories(categories):
        if categories is None:
            return ['train', 'val', 'test']
        valid_categories = {'train', 'val', 'test'}
        categories = list(categories)
        for category in categories:
            if category not in valid_categories:
                raise ValueError(f"Unknown data category: {category}.")
        return categories


    @staticmethod
    def _build_extra_inputs(x, timestamps):
        return {
            'timestamps': timestamps,
            'x_missing_mask': (x[..., :1] == 0)
        }


    def _select_value_channel(self, x, y):
        if not self.value_only:
            return x, y
        # STBoost only needs the traffic value channel after frozen base predictions are cached.
        return x[..., :1].copy(), y[..., :1].copy()


    def _preprocess_data(self, stage):
        if self.base_model != 'MegaCRN':
            raise ValueError('This release supports MegaCRN only')
        if self.is_enhanced:
            self.logger.info(f"====== Generating stage {stage} training data! ==========")
        else:
            self.logger.info("====== Generating training data! ==========")
        if stage == 1:
            self.p = self.args.seq_len
            self.q = self.args.horizon
        else:
            self.p = self.args.p_len
            self.q = self.args.q_len
        return self._preprocess_MegaCRN(full_timestamps=self.full_timestamps)


    def _unified_load_dataset(self):
        if self.base_model != 'MegaCRN':
            raise ValueError('This release supports MegaCRN only')
        self.logger.info("Initial Dataset Splitting")
        df = pd.read_hdf(self.args.traffic_df_filename)
        self.logger.info(f"original dataset shape:  {df.shape}")

        # divide the three dataset first to avoid data leakage.
        total_steps = len(df)
        test_ratio = 1 - self.args.trainval_ratio
        train_ratio = self.args.trainval_ratio * (1 - self.args.val_ratio)
        val_ratio = 1 - train_ratio - test_ratio
        train_end = round(total_steps * train_ratio)
        val_end = round(total_steps * (train_ratio + val_ratio))

        df_train = df.iloc[: train_end]
        self.logger.info(f"df_train shape: {df_train.shape}")
        df_val = df.iloc[train_end: val_end]
        self.logger.info(f"df_val shape: {df_val.shape}")
        df_test = df.iloc[val_end:]
        self.logger.info(f"df_test shape: {df_test.shape}")
        return {'train': df_train, 'val': df_val, 'test': df_test}


    @staticmethod
    def _generate_graph_seq2seq_io_data(
            df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None,
            full_timestamps=False
    ):
        """
        Generate samples from
        :param df:
        :param x_offsets:
        :param y_offsets:
        :param add_time_in_day:
        :param add_day_in_week:
        :param scaler:
        :return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """

        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values.astype(np.float32), axis=-1)       # convert to float32 to reduce the VRAM cost
        data_list = [data]
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)  # reduce the VRAM cost
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week.astype(np.float32))    # reduce the VRAM cost

        data = np.concatenate(data_list, axis=-1)
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

        # rewrite to reduce memory consumption
        num_samples = max_t - min_t
        x_shape = (num_samples, len(x_offsets), num_nodes, data.shape[-1])
        y_shape = (num_samples, len(y_offsets), num_nodes, data.shape[-1])
        timestamps_shape = (num_samples, len(x_offsets) + len(y_offsets)) if full_timestamps else (num_samples,)

        x = np.empty(x_shape, dtype=np.float32)
        y = np.empty(y_shape, dtype=np.float32)
        timestamps = np.empty(timestamps_shape, dtype='datetime64[s]')
        if full_timestamps:
            time_offsets = np.concatenate([x_offsets, y_offsets])

        for i, t in enumerate(range(min_t, max_t)):
            x[i] = data[t + x_offsets, ...]
            y[i] = data[t + y_offsets, ...]
            if full_timestamps:
                timestamps[i] = df.index[t + time_offsets].to_numpy()
            else:
                timestamps[i] = df.index[t + x_offsets[0]]

        return x, y, timestamps


    def _preprocess_MegaCRN(self, full_timestamps=False):
        """

        :return: data: dict
        data = {
            'x_train': numpy.float32, (),
            'y_train': numpy.float32, (),
            'x_val': numpy.float32, (),
            'y_val': numpy.float32, (),
            'x_test': numpy.float32, (),
            'y_test': numpy.float32, (),
            'x_offsets': numpy
            'y_offsets': numpy
        }
        """
        data_frames = {
            'train': self.df['train'],
            'val': self.df['val'],
            'test': self.df['test']
        }

        # 0 is the latest observed sample.
        x_offsets = np.sort(
            np.concatenate((np.arange(-(self.p - 1), 1, 1),))  # Modify to support variable seq_len
        )
        y_offsets = np.sort(np.arange(1, (self.q + 1), 1))     # Modify to support variable horizon
        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)

        data = {}
        for cat in self.active_categories:
            _x, _y, _timestamps = self._generate_graph_seq2seq_io_data(
                data_frames[cat], x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False,
                full_timestamps=full_timestamps
            )
            _x, _y = self._select_value_channel(_x, _y)
            self.logger.info(f"{cat} x:  {_x.shape} y:  {_y.shape}")
            data['x_' + cat] = _x
            data['y_' + cat] = _y
            data['extra_inputs_' + cat] = self._build_extra_inputs(_x, _timestamps)
        data['x_offsets'] = x_offsets.reshape(list(x_offsets.shape) + [1])
        data['y_offsets'] = y_offsets.reshape(list(y_offsets.shape) + [1])
        return data


    def _process_STBoost_adj(self):
        cache_path = f"./datasets/{self.args.dataset}/{self.args.dataset}_stboost_sd_mx.npy"
        if os.path.exists(cache_path):
            self.logger.info(f"Loading STBoost sd_mx cache file {cache_path}")
            sd_mx_cached = np.load(cache_path)
            expected_shape = (self.args.num_nodes, self.args.num_nodes)
            if sd_mx_cached.shape == expected_shape:
                return sd_mx_cached.astype(np.float32)
            self.logger.info(
                f"STBoost sd_mx cache shape mismatch: expected {expected_shape}, got {sd_mx_cached.shape}. Recomputing."
            )

        # load geo
        geofile = pd.read_csv(f"./datasets/{self.args.dataset}/sensor_id_{self.args.dataset}.txt", header=None)
        geo_ids = geofile.values[0].tolist()
        num_nodes = len(geo_ids)
        geo_to_ind = {}
        for index, idx in enumerate(geo_ids):
            geo_to_ind[idx] = index
        self.logger.info(f"Loaded file ./datasets/{self.args.dataset}/sensor_id_{self.args.dataset}.txt" + ', num_nodes=' + str(num_nodes))

        # load rel
        if self.args.dataset in ['PEMS03', 'PEMS08']:
            rel_path = f"./datasets/{self.args.dataset}/{self.args.dataset}.csv"
        else:
            raise NotImplementedError("Unsupported dataset")
        relfile = pd.read_csv(rel_path, dtype={'from': 'int', 'to': 'int'})
        adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        # leverage dist matrix
        adj_mx[:] = np.inf
        for row in relfile.values:
            if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
                continue
            # directed graph
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = row[2]
        self.logger.info("Loaded file " + rel_path + ', shape=' + str(adj_mx.shape))
        self.logger.info('Max adj_mx value = {}'.format(adj_mx.max()))

        self.logger.info("Start Calculate the weight by Gauss kernel!")
        sd_mx = adj_mx.copy()
        distances = adj_mx[~np.isinf(adj_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(adj_mx / std))
        # base on distance
        sd_mx[adj_mx == 0] = np.inf
        for k in range(self.args.num_nodes):
            for i in range(self.args.num_nodes):
                for j in range(self.args.num_nodes):
                    sd_mx[i, j] = min(sd_mx[i, j], sd_mx[i, k] + sd_mx[k, j])

        sd_mx = sd_mx.astype(np.float32)
        self.logger.info(f"Saving STBoost sd_mx cache file {cache_path}")
        np.save(cache_path, sd_mx)
        return sd_mx
