"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14 20:15
# @Author  : Chen Wang
# @Site    : 
# @File    : STBoost_trainer.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import hashlib
import json
import os
import uuid
import torch
import datetime
import gc
import time
import numpy as np
from utils.logger import get_logger
from models.STBoost.TimeInterpreter import TimeInterpreter
from models.STBoost.PeriodPatternMemorizer import PeriodPatternMemorizer
from utils.metrics import evaluate_all_metrics
from utils.tools import save_rng_state, restore_rng_state


class STBoost_Trainer:
    def __init__(self, args, base_seed):
        self.args = args
        self.base_seed = base_seed
        self.save_path = self._create_unique_save_path()

        # Convert start and end date strings to datetime64[D]
        assert self.args.start_date_str is not None, "Please specify the start date of dataset."
        assert self.args.end_date_str is not None, "Please specify the end date of dataset."
        self.start_date = np.datetime64(self.args.start_date_str, 'D')
        self.end_date = np.datetime64(self.args.end_date_str, 'D')
        self.args.num_days = (self.end_date - self.start_date).astype('timedelta64[D]').astype(int) + 1 # Include both ends

        cpu_num = 1
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)

        self.logger = self._set_logger()

        self.best_model_path = f"{self.save_path}/best_model.pt"

    def _create_unique_save_path(self):
        for _ in range(16):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.timestr = f'{timestamp}-{uuid.uuid4().hex[:8]}'
            save_path = self._build_save_path()
            try:
                os.makedirs(save_path, exist_ok=False)
                return save_path
            except FileExistsError:
                continue
        raise RuntimeError('Failed to create a unique experiment directory.')

    @staticmethod
    def _safe_exp_segment(value):
        value = str(value).strip()
        for old, new in [('/', '_'), ('\\', '_'), (' ', '_')]:
            value = value.replace(old, new)
        return value

    def _build_save_path(self):
        root = f'./experiments/{self.args.base_model}_{self.args.is_enhanced}'
        if self.args.online_inference_only:
            root = f'./experiments/online_inference/{self.args.base_model}_{self.args.is_enhanced}'
        category = self._safe_exp_segment(getattr(self.args, 'stb_exp_category', ''))
        exp_name = self._safe_exp_segment(getattr(self.args, 'stb_exp_name', ''))

        if category and exp_name:
            return f'{root}/{category}/{exp_name}/{self.args.dataset}/{self.args.p_len}-{self.args.q_len}/{self.timestr}'
        if category:
            return f'{root}/{category}/{self.args.dataset}/{self.args.p_len}-{self.args.q_len}/{self.timestr}'
        return f'{root}/{self.args.dataset}/{self.args.p_len}-{self.args.q_len}/{self.timestr}'

    def _stage2_can_use_value_only_data(self):
        if not self.args.is_enhanced:
            return False
        return (not self.args.stb_use_base_prediction) or bool(self.args.stb_cache_base_prediction)

    def _stage2_should_generate_value_only_data(self, stage):
        return stage == 2 and self._stage2_can_use_value_only_data() and not self.args.stb_use_base_prediction

    def _use_stage2_inference_cache(self):
        return (
            bool(self.args.is_enhanced)
            and bool(self.args.stb_use_base_prediction)
            and bool(self.args.stb_cache_base_prediction)
            and bool(self.args.stb_use_inference_cache)
        )

    def _sha256_file(self, path):
        if not path or not os.path.isfile(path):
            return None
        if not hasattr(self, '_sha256_cache'):
            self._sha256_cache = {}
        path = os.path.abspath(path)
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key not in self._sha256_cache:
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            self._sha256_cache[key] = digest.hexdigest()
        return self._sha256_cache[key]

    def _inference_cache_paths(self, mode, base_p_len, base_q_len):
        cache_dir = os.path.join(self.args.stb_inference_cache_root, self.args.base_model, self.args.dataset)
        checkpoint_hash = self._sha256_file(getattr(self, 'pretrained_path', ''))
        name = (
            f'{mode}_seed{self.base_seed}_stage2-{self.args.p_len}-{self.args.q_len}'
            f'_base-{base_p_len}-{base_q_len}_batch-{self.args.base_batch_size}'
            f'_checkpoint-{checkpoint_hash[:12]}'
        )
        if getattr(self.args, 'stb_stage2_align_inputs', 0):
            name += '_aligned-inputs-affine-v1'
        return cache_dir, os.path.join(cache_dir, name + '.npy'), os.path.join(cache_dir, name + '.json')

    def _inference_cache_metadata(self, mode, loader, base_p_len, base_q_len):
        pretrained_path = getattr(self, 'pretrained_path', '')
        checkpoint_hash = self._sha256_file(pretrained_path)
        if checkpoint_hash is None:
            return None
        output_dim = int(getattr(self.args, 'output_dim', 1))
        metadata = {
            'base_model': self.args.base_model,
            'dataset': self.args.dataset,
            'mode': mode,
            'seed': int(self.base_seed),
            'p_len': int(self.args.p_len),
            'q_len': int(self.args.q_len),
            'base_p_len': int(base_p_len),
            'base_q_len': int(base_q_len),
            'base_batch_size': int(self.args.base_batch_size),
            'pretrained_path': os.path.normpath(pretrained_path),
            'pretrained_sha256': checkpoint_hash,
            'original_size': int(loader.original_size),
            'num_nodes': int(self.args.num_nodes),
            'output_dim': output_dim,
            'shape': [int(loader.original_size), int(base_q_len), int(self.args.num_nodes), output_dim],
            'dtype': 'float32',
        }
        if getattr(self.args, 'stb_stage2_align_inputs', 0):
            metadata.update({
                'stage2_input_alignment': 1,
                'base_scaler_bridge': 'affine-v1',
                'base_scaler': [float(self.base_scaler.mean), float(self.base_scaler.std)],
                'stage2_scaler': [float(self.scaler.mean), float(self.scaler.std)],
            })
        return metadata

    def _load_stage2_inference_cache(self, mode, loader, base_p_len, base_q_len):
        if not self._use_stage2_inference_cache():
            return None
        metadata = self._inference_cache_metadata(mode, loader, base_p_len, base_q_len)
        if metadata is None:
            self.logger.info(f'Skip persistent inference cache for {mode}: checkpoint unavailable.')
            return None
        _, npy_path, json_path = self._inference_cache_paths(mode, base_p_len, base_q_len)
        if not (os.path.isfile(npy_path) and os.path.isfile(json_path)):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as handle:
                cached_metadata = json.load(handle)
            if not isinstance(cached_metadata, dict):
                self.logger.info(f'Ignore invalid inference cache metadata for {mode}.')
                return None
            mismatch = next((key for key, value in metadata.items()
                             if cached_metadata.get(key) != value), None)
            if mismatch is not None:
                self.logger.info(f'Ignore stale inference cache for {mode}: {mismatch} mismatch.')
                return None
            cache = np.load(npy_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.logger.info(f'Ignore unreadable inference cache for {mode}: {exc}.')
            return None
        if list(cache.shape) != metadata['shape'] or cache.dtype != np.float32:
            self.logger.info(f'Ignore invalid inference cache for {mode}: shape or dtype mismatch.')
            return None
        cache = np.ascontiguousarray(cache)
        loader.extra_inputs['y_base_cache'] = cache
        self.logger.info(f'Loaded persistent inference cache for {mode}: {npy_path}, shape={cache.shape}.')
        return cache

    def _save_stage2_inference_cache(self, mode, loader, cache, base_p_len, base_q_len):
        if not self._use_stage2_inference_cache():
            return
        metadata = self._inference_cache_metadata(mode, loader, base_p_len, base_q_len)
        if metadata is None:
            self.logger.info(f'Skip saving persistent inference cache for {mode}: checkpoint unavailable.')
            return
        if list(cache.shape) != metadata['shape'] or cache.dtype != np.float32:
            raise RuntimeError(f'Invalid inference cache for {mode}: shape or dtype mismatch.')
        cache_dir, npy_path, json_path = self._inference_cache_paths(mode, base_p_len, base_q_len)
        os.makedirs(cache_dir, exist_ok=True)
        temp_id = uuid.uuid4().hex
        tmp_npy = npy_path + f'.{temp_id}.tmp.npy'
        tmp_json = json_path + f'.{temp_id}.tmp'
        try:
            np.save(tmp_npy, cache)
            with open(tmp_json, 'w', encoding='utf-8') as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write('\n')
            os.replace(tmp_npy, npy_path)
            os.replace(tmp_json, json_path)
        finally:
            for temp_path in (tmp_npy, tmp_json):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        self.logger.info(f'Saved persistent inference cache for {mode}: {npy_path}.')

    def _release_data_loaders(self, modes):
        released = False
        for mode in modes:
            if self.data.pop(f'{mode}_loader', None) is not None:
                released = True
                self.logger.info(f"Released {mode} dataloader from memory.")
        if released:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _slim_stage2_value_channels(self, modes):
        if not self._stage2_can_use_value_only_data():
            return

        released = False
        for mode in modes:
            loader = self.data.get(f'{mode}_loader')
            if loader is None:
                continue
            if loader.xs.shape[-1] == 1 and loader.ys.shape[-1] == 1:
                continue

            # STBoost only consumes the target value channel once frozen base predictions are cached.
            # Keep x_missing_mask unchanged because it was built from raw values before normalization.
            loader.xs = np.ascontiguousarray(loader.xs[..., :1])
            loader.ys = np.ascontiguousarray(loader.ys[..., :1])
            released = True
            self.logger.info(
                f"Slimmed {mode} stage-2 data to value channels: "
                f"x {loader.xs.shape}, y {loader.ys.shape}."
            )
        if released:
            gc.collect()

    def _init_time_interpreter(self, holiday_csv_path, device):
        time_interpreter = TimeInterpreter(
            path=holiday_csv_path,
            p_len=self.args.p_len,
            q_len=self.args.q_len,
            device=device,
            duration=self.args.duration,
            start_date=self.start_date
        )
        time_interpreter.reset_timestamps()  # init
        return time_interpreter

    def _print_model(self, model):
        param_count = 0
        self.logger.info('Trainable parameter list:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.logger.info(f"{name} {param.shape} {param.numel()}")
                param_count += param.numel()
        self.logger.info(f'In total: {param_count} trainable parameters.')

    def _set_logger(self, log_path=None):
        name = '{}_{}_{}'.format(self.args.base_model,
                                 self.args.dataset,
                                 self.timestr)
        if log_path is None:
            logger = get_logger(root=self.save_path, name=name, is_debug_in_screen=True)
        else:
            logger = get_logger(root=self.save_path, name=name, is_debug_in_screen=True, log_path=log_path)
        return logger

    def _load_pretrained_enhanced_model(self):
        """Cold-load a complete checkpoint for evaluation without training either stage."""
        checkpoint_path = self.args.best_model_path
        if checkpoint_path is None:
            assert self.args.best_model_timestamps is not None, 'best_model_timestamps must be set!'
            checkpoint_dir = os.path.join('./experiments', f'{self.args.base_model}_{self.args.is_enhanced}')
            category = self._safe_exp_segment(getattr(self.args, 'stb_exp_category', ''))
            exp_name = self._safe_exp_segment(getattr(self.args, 'stb_exp_name', ''))
            if category:
                checkpoint_dir = os.path.join(checkpoint_dir, category)
                if exp_name:
                    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
            checkpoint_path = os.path.join(
                checkpoint_dir, self.args.dataset, f'{self.args.p_len}-{self.args.q_len}',
                self.args.best_model_timestamps, 'best_model.pt'
            )
        assert os.path.isfile(checkpoint_path), f'best model path {checkpoint_path} does not exist!'

        rng_state = save_rng_state(self.device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self._init_data(stage=1, categories=('train', 'val'))
            model = self._get_model()

            if self.args.is_enhanced:
                self._release_data_loaders(('train', 'val'))
                self._init_data(stage=2)
                if self.args.stb_stage2_align_inputs:
                    model.base_scaler = self.base_scaler
                    model.stage2_scaler = self.scaler
                if model.booster.use_period_memorizer:
                    prefix = 'booster.period_memorizer.'
                    memory = state_dict[prefix + 'temp_pattern_memory']
                    expected_shape = (
                        self.args.stb_num_period_pattern, (24 * 60) // self.args.duration,
                        self.args.num_nodes, self.args.stb_period_mem_dim
                    )
                    if tuple(memory.shape) != expected_shape or memory.shape[-1] != 1:
                        raise ValueError(f'PPM checkpoint shape {tuple(memory.shape)} does not match {expected_shape}.')
                    ppm = PeriodPatternMemorizer(
                        num_pattern=self.args.stb_num_period_pattern,
                        num_sensors=self.args.num_nodes,
                        period_mem_dim=self.args.stb_period_mem_dim,
                        day_slots=(24 * 60) // self.args.duration,
                        init_values=memory.squeeze(-1).detach().cpu().numpy()
                    ).to(self.device)
                    ppm.fix_pattern_memory()
                    # Register the saved retrieval buffer without rescaling its values.
                    if prefix + 'retrieval_memory' in state_dict:
                        ppm.register_buffer('retrieval_memory', torch.zeros_like(state_dict[prefix + 'retrieval_memory']))
                    model.set_period_pattern_memorizer(ppm)

            model.load_state_dict(state_dict)
            if self.args.is_enhanced:
                model.set_stage(2)
                model.set_unified_training()
            model.eval()
            return model
        finally:
            restore_rng_state(rng_state, self.device)

    def _online_inference_forward(self, model, x, y, extra_inputs, stage):
        """Keep the model-specific forward identical to evaluation, without loss or caches."""
        base_model = self.args.base_model
        if base_model == 'MegaCRN':
            x, y, ycov = self._prepare_x_y(x, y)
        else:
            raise ValueError(f'Unsupported online inference model: {base_model}')

        if stage == 1:
            if base_model == 'MegaCRN':
                output, *_ = model(x, ycov)
            else:
                raise ValueError(f'Unsupported online inference model: {base_model}')
        else:
            mask = extra_inputs.get('x_missing_mask')
            mask = torch.from_numpy(mask).bool().to(self.device) if mask is not None else None
            self.time_interpreter.set_timestamps(extra_inputs['timestamps'])
            encoding = self.time_interpreter.get_time_encoding()
            info = self.time_interpreter.get_time_info(self.args.stb_num_period_pattern)
            if base_model == 'MegaCRN':
                output = model(x, ycov, encoding, info, self.boost_dist_adj,
                               x_missing_mask=mask, y_base_cache=None)
            else:
                raise ValueError(f'Unsupported online inference model: {base_model}')
        return self.scaler.inverse_transform(output).detach().cpu()

    def profile_online_inference(self, model, stage, checkpoint_path=None):
        """Time one uncached test pass, retaining the true final partial batch.

        The timed scope is host-array batching, input transfer/calendar encoding,
        full model forward, inverse scaling and CPU prediction transfer. It
        excludes loading/preprocessing data, checkpoint loading, warm-up, losses,
        metrics and file writes. Per-sample time is batch-amortized, not B=1 latency.
        """
        if not self.args.profile_online_inference:
            return None
        if stage not in (1, 2) or (stage == 2) != bool(self.args.is_enhanced):
            raise ValueError('Profile Direct at stage 1 or STBoost at stage 2.')
        if stage == 2 and not model.use_base_prediction:
            self.logger.info('Skip full-system online timing: this ablation disables the backbone prediction.')
            return None
        loader = self.data['test_loader']
        size = loader.original_size
        batch_size = self.args.online_inference_batch_size
        if size <= 0 or batch_size <= 0:
            raise ValueError('Online inference needs nonempty test data and a positive batch size.')
        required_x = getattr(self.args, 'input_dim', 1)
        required_y = 2 * self.args.output_dim
        if loader.xs.shape[-1] < required_x or loader.ys.shape[-1] < required_y:
            raise ValueError('Profile before slimming value/calendar channels or reload the full test data.')
        report_path = os.path.join(self.save_path, 'online_inference.json')
        if os.path.exists(report_path):
            raise FileExistsError(f'Online inference report already exists: {report_path}')
        timestamps = loader.extra_inputs['timestamps'][:size]
        starts = timestamps[:, 0] if timestamps.ndim == 2 else timestamps
        sample_hash = hashlib.sha256(np.asarray(starts, dtype='datetime64[ns]').tobytes()).hexdigest()
        checkpoint_path = checkpoint_path or self.best_model_path
        checkpoint_hash = self._sha256_file(checkpoint_path)
        if checkpoint_hash is None:
            raise FileNotFoundError(f'Online inference checkpoint is missing: {checkpoint_path}')
        state_rng = save_rng_state(self.device)
        training_modes = [(module, module.training) for module in model.modules()]
        time_state = (self.time_interpreter.timestamps, self.time_interpreter.time_encoding)
        batches = (size + batch_size - 1) // batch_size
        warmup_batches = 3

        def forward_batch(start):
            end = min(start + batch_size, size)
            extras = {key: values[start:end] for key, values in loader.extra_inputs.items()
                      if key != 'y_base_cache'}
            self._online_inference_forward(model, loader.xs[start:end], loader.ys[start:end], extras, stage)

        def synchronize():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
            if stage == 2:
                if not model.get_unified_training():
                    self.logger.info('Skip full-system online timing: unified gated output is not active.')
                    return None
            model.eval()
            with torch.no_grad():
                for _ in range(warmup_batches):
                    forward_batch(0)
                synchronize()
                start_time = time.perf_counter()
                for start in range(0, size, batch_size):
                    forward_batch(start)
                synchronize()
                total = time.perf_counter() - start_time
        finally:
            for module, was_training in training_modes:
                module.training = was_training
            self.time_interpreter.timestamps, self.time_interpreter.time_encoding = time_state
            restore_rng_state(state_rng, self.device)

        record = {
            'version': 1, 'method': 'STBoost' if stage == 2 else 'Direct',
            'base_model': self.args.base_model, 'dataset': self.args.dataset,
            'input_length': int(loader.xs.shape[1]), 'horizon': int(loader.ys.shape[1]),
            'batch_size': batch_size, 'samples': size, 'batches': batches,
            'sample_unit': 'forecast_window',
            'warmup_batches': warmup_batches, 'total_s': total,
            'mean_batch_ms': total * 1000 / batches, 'amortized_sample_ms': total * 1000 / size,
            'scope': 'host_batch_to_cpu_prediction_v1', 'prediction_cache': False,
            'sample_fingerprint': sample_hash,
            'device_type': self.device.type,
            'device_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
            'precision': str(next(model.parameters()).dtype),
            'checkpoint_path': os.path.abspath(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
        }
        with open(report_path, 'x', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
            f.write('\n')
        self.logger.info('[Online Inference] ' + json.dumps(record, sort_keys=True))
        self.logger.info(
            f'Online inference completed: total={total:.6f}s, '
            f'mean_batch={record["mean_batch_ms"]:.6f}ms, '
            f'amortized_sample={record["amortized_sample_ms"]:.6f}ms; '
            f'{size} samples / {batches} batches, batch_size={batch_size}, no prediction cache.'
        )
        return record

    def test_from_checkpoint(self):
        """Evaluate saved Direct/STBoost weights and time uncached online inference."""
        if not self.args.best_model_path:
            raise ValueError('--test_only requires --best_model_path.')
        if not self.args.profile_online_inference:
            raise ValueError('Enable --profile_online_inference for checkpoint-only testing.')
        model = self._load_pretrained_enhanced_model()
        stage = 2 if self.args.is_enhanced else 1
        if stage == 1:
            self._init_data(stage=1, categories=('test',))
        self._release_data_loaders(('train', 'val'))
        self.data['test_loader'].extra_inputs.pop('y_base_cache', None)
        inference = self.profile_online_inference(model, stage, self.args.best_model_path)

        self.logger.info('=' * 35 + 'Checkpoint-only test performance' + '=' * 35)
        evaluation_loader = self.data['test_loader']
        self.logger.info(
            f'Checkpoint-only metric evaluation batch_size: {evaluation_loader.batch_size}; '
            'using the original stage batch/indices and storing predictions on CPU.'
        )
        try:
            if stage == 2 and model.use_base_prediction:
                # Match normal Stage 2 evaluation: run the frozen backbone at its
                # native batch size, then evaluate Sidecar at the STBoost batch size.
                # This temporary metric cache is built AFTER uncached timing.
                self.logger.info('Building temporary in-memory base predictions for metrics only; excluded from online timing.')
                state_rng = save_rng_state(self.device)
                use_disk_cache = self.args.stb_use_inference_cache
                try:
                    self.args.stb_use_inference_cache = 0
                    self._build_base_prediction_cache_for_loader(model.base_model, evaluation_loader, 'test')
                finally:
                    self.args.stb_use_inference_cache = use_disk_cache
                    restore_rng_state(state_rng, self.device)
            test_loss, y_true, y_pred = self._evaluate(model, 'test', stage, store_on_cpu=True)
        finally:
            evaluation_loader.extra_inputs.pop('y_base_cache', None)
        if self.args.base_model == 'MegaCRN':
            # MegaCRN evaluation returns (Q, B, S, D).
            pred = np.transpose(y_pred[..., 0], (1, 2, 0))
            true = np.transpose(y_true[..., 0], (1, 2, 0))
        else:
            raise ValueError(f'Unsupported checkpoint evaluation model: {self.args.base_model}')
        maes, rmses, mapes = evaluate_all_metrics(
            pred, true, 0., self.logger, use_adaptive_mask=(stage == 2)
        )
        metrics = {'mae': float(np.mean(maes)), 'rmse': float(np.mean(rmses)),
                   'mape': float(np.mean(mapes)),
                   'per_horizon': {'mae': list(map(float, maes)), 'rmse': list(map(float, rmses)),
                                   'mape': list(map(float, mapes))}}
        self.logger.info('Checkpoint-only test completed; no training was performed.')
        return {'test_loss': float(test_loss), 'metrics': metrics, 'inference': inference}

    def run_online_inference_only(self):
        """Compatibility entry point for --online_inference_only and --test_only."""
        return self.test_from_checkpoint()

    def _print_share_args(self):
        self.logger.info('========== Shared args: ==========')
        self.logger.info('base_seed:', self.base_seed)
        self.logger.info('dataset:', self.args.dataset)
        self.logger.info('duration:', self.args.duration)
        self.logger.info('num_nodes:', self.args.num_nodes)
        self.logger.info('base_model:', self.args.base_model)
        self.logger.info('is_enhanced:', self.args.is_enhanced)
        self.logger.info('profile_online_inference:', self.args.profile_online_inference)
        self.logger.info('online_inference_batch_size:', self.args.online_inference_batch_size)
        self.logger.info('online_inference_only:', self.args.online_inference_only)
        self.logger.info('dataset start date:', self.start_date)
        self.logger.info('dataset end date:', self.end_date)
        self.logger.info('num_days:', self.args.num_days)

    def _print_STBoost_args(self):
        # print STBoost params
        if self.args.is_enhanced:
            self.logger.info('========== STBoost args: ==========')
            self.logger.info('stb_init_seed:', self.args.stb_init_seed)
            self.logger.info('p_len:', self.args.p_len)
            self.logger.info('q_len:', self.args.q_len)
            self.logger.info('stb_dynamic_spatial_embed_dim:', self.args.stb_dynamic_spatial_embed_dim)
            self.logger.info('stb_num_hyper_nodes:', self.args.stb_num_hyper_nodes)
            self.logger.info('stb_dynamic_cheb_k:', self.args.stb_dynamic_cheb_k)
            self.logger.info('stb_time_embed_dim:', self.args.stb_time_embed_dim)
            self.logger.info('stb_num_period_pattern:', self.args.stb_num_period_pattern)
            self.logger.info('stb_period_mem_dim:', self.args.stb_period_mem_dim)
            self.logger.info('stb_static_spatial_embed_dim:', self.args.stb_static_spatial_embed_dim)
            self.logger.info('stb_static_cheb_k:', self.args.stb_static_cheb_k)
            self.logger.info('stb_static_hid_dim:', self.args.stb_static_hid_dim)
            self.logger.info('stb_temporal_embed_dim:', self.args.stb_temporal_embed_dim)
            self.logger.info('stb_st_mixer_depth:', self.args.stb_st_mixer_depth)
            self.logger.info('stb_temporal_dropout:', self.args.stb_temporal_dropout)
            self.logger.info('stb_use_input_temporal:', bool(self.args.stb_use_input_temporal))
            self.logger.info('stb_use_input_period:', bool(self.args.stb_use_input_period))
            self.logger.info('stb_use_input_residual:', bool(self.args.stb_use_input_residual))
            self.logger.info('stb_use_period_memorizer:', bool(self.args.stb_use_period_memorizer))
            self.logger.info('stb_ppm_mode:', self.args.stb_ppm_mode)
            self.logger.info('stb_ppm_align_scaler:', self.args.stb_ppm_align_scaler)
            self.logger.info('stb_ppm_pad_pairs:', self.args.stb_ppm_pad_pairs)
            self.logger.info('stb_ppm_reshuffle_each_epoch:', self.args.stb_ppm_reshuffle_each_epoch)
            self.logger.info('stb_ppm_control_rng:', self.args.stb_ppm_control_rng)
            self.logger.info('stb_stage2_align_inputs:', self.args.stb_stage2_align_inputs)
            self.logger.info('stb_sidecar_mode:', self.args.stb_sidecar_mode)
            self.logger.info('stb_mlp_hidden_dim:', self.args.stb_mlp_hidden_dim)
            self.logger.info('stb_use_sample_wise_adaptive_graph:', bool(self.args.stb_use_sample_wise_adaptive_graph))
            self.logger.info('stb_use_static_graph:', bool(self.args.stb_use_static_graph))
            self.logger.info('stb_use_spatial_modeling:', bool(self.args.stb_use_spatial_modeling))
            self.logger.info('stb_use_base_prediction:', bool(self.args.stb_use_base_prediction))
            self.logger.info('stb_cache_base_prediction:', bool(self.args.stb_cache_base_prediction))
            self.logger.info('stb_use_inference_cache:', bool(self.args.stb_use_inference_cache))
            self.logger.info('stb_inference_cache_root:', self.args.stb_inference_cache_root)
            self.logger.info('stb_batch_size:', self.args.stb_batch_size)

            self.logger.info('stb_use_pretrained:', self.args.stb_use_pretrained)
            self.logger.info('stb_is_unified_first:', self.args.stb_is_unified_first)
            self.logger.info('stb_unified_switch_epoch:', self.args.stb_unified_switch_epoch)
            self.logger.info('stb_epochs:', self.args.stb_epochs)
            self.logger.info('stb_base_lr:', self.args.stb_base_lr)
            self.logger.info('stb_lr_milestones:', self.args.stb_lr_milestones)
            self.logger.info('stb_patience:', self.args.stb_patience)
            self.logger.info('stb_use_curriculum_learning:', self.args.stb_use_curriculum_learning)
            self.logger.info('stb_step_size:', self.args.stb_step_size)
            self.logger.info('stb_reach_frac:', self.args.stb_reach_frac)
            self.logger.info('stb_unified_init_logit:', self.args.stb_unified_init_logit)
            self.logger.info('stb_lamb_mse:', self.args.stb_lamb_mse)
            self.logger.info('stb_use_tod:', bool(self.args.stb_use_tod))
            self.logger.info('stb_use_dow:', bool(self.args.stb_use_dow))
            self.logger.info('stb_dump_debug_arrays:', self.args.stb_dump_debug_arrays)
