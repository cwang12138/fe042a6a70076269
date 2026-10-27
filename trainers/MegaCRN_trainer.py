"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 22:49
# @Author  : Chen Wang
# @Site    : 
# @File    : MegaCRN_trainer.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import os
import gc
import hashlib
import torch
import torch.nn as nn
import time
import math
import numpy as np
from trainers.STBoost_trainer import STBoost_Trainer
from models.STBoost.STBoost import STBoost, ParallelStructure
from models.STBoost.PeriodPatternMemorizer import PeriodPatternMemorizer, SelfSupervisedDataLoader, train_ssl_general_pattern
from models.MegaCRN.utils import StandardScaler, DataLoader
from models.MegaCRN.MegaCRN import MegaCRN
from utils.data_factory import Data_Traffic
from utils.metrics import (
    masked_mae_loss,
    masked_mse_loss,
    evaluate_all_metrics,
    set_adaptive_mask_epsilon,
)
from utils.tools import memory_report, save_rng_state, restore_rng_state, convert_standardized_scale


class MegaCRN_Trainer(STBoost_Trainer):
    def __init__(self, args, base_seed):
        super().__init__(args, base_seed)

        self._print_share_args()
        self._print_MegaCRN_args()
        self._print_STBoost_args()

        self.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)


        self.data = {}
        self.data_factory = Data_Traffic(self.args, 'MegaCRN', self.logger, self.args.is_enhanced)

        if self.args.is_enhanced:
            # prepare STBoost data
            self.train_data = self.data_factory.get_train_dataset()
            self.boost_dist_adj = torch.from_numpy(self.data_factory.get_boost_adj()).float().to(self.device)

            pretrained_filename = f"{self.args.base_model}_{self.args.dataset}_{self.args.seq_len}_{self.args.horizon}_{self.args.base_batch_size}_{base_seed}_base.pt"
            if self.args.stb_use_pretrained:
                # check pretrained path
                pretrained_root_path = f"./experiments/pretrained/{self.args.base_model}"
                if not os.path.exists(pretrained_root_path):
                    os.makedirs(pretrained_root_path)
                self.pretrained_path = f"{pretrained_root_path}/{pretrained_filename}"
            else:
                # Keep explicitly retrained Stage 1 weights with their own experiment log.
                self.pretrained_path = f"{self.save_path}/{pretrained_filename}"

        self.holiday_csv_path = self.args.traffic_df_filename[
                                : self.args.traffic_df_filename.rfind('/')] + "/holidays.csv"
        self.time_interpreter = self._init_time_interpreter(self.holiday_csv_path, self.device)

    def _init_data(self, stage, categories=None):
        categories = ['train', 'val', 'test'] if categories is None else list(categories)
        data = self.data_factory.get_data(
            stage,
            categories=categories,
            value_only=self._stage2_should_generate_value_only_data(stage)
        )
        if 'train' in categories:
            self.scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
            if stage == 1 and self.args.is_enhanced and self.args.stb_stage2_align_inputs:
                self.base_scaler = self.scaler
            mask_eps = set_adaptive_mask_epsilon(self.scaler.mean, self.scaler.std)
        else:
            assert hasattr(self, 'scaler'), "Scaler must be initialized from train data before loading test-only data."
            mask_eps = set_adaptive_mask_epsilon(self.scaler.mean, self.scaler.std)
        for category in categories:
            data['x_' + category][..., 0] = self.scaler.transform(data['x_' + category][..., 0])
            if category != 'test':
                data['y_' + category][..., 0] = self.scaler.transform(data['y_' + category][..., 0])
        for category in categories:
            self.logger.info(f"{category} xs.shape, ys.shape", data['x_' + category].shape, data['y_' + category].shape)
        self.logger.info(f'Adaptive mask epsilon from scaler round-trip zero: {mask_eps:.12g}')
        stage_batch_size = self.args.base_batch_size
        if stage == 2 and self.args.is_enhanced and self.args.stb_batch_size is not None:
            stage_batch_size = self.args.stb_batch_size
        self.logger.info(f'MegaCRN stage {stage} dataloader batch_size: {stage_batch_size}')
        for category in categories:
            self.data[f'{category}_loader'] = DataLoader(
                data['x_' + category], data['y_' + category], data['extra_inputs_' + category],
                stage_batch_size, shuffle=(category == 'train')
            )

    def _get_model(self):
        base_model = MegaCRN(
            num_nodes=self.args.num_nodes,
            input_dim=self.args.input_dim,
            output_dim=self.args.output_dim,
            ycov_dim=self.args.output_dim,
            horizon=self.args.horizon,
            rnn_units=self.args.rnn_units,
            num_layers=self.args.num_rnn_layers,
            mem_num=self.args.mem_num,
            mem_dim=self.args.mem_dim,
            cheb_k=self.args.max_diffusion_step,
            cl_decay_steps=self.args.cl_decay_steps,
            use_curriculum_learning=self.args.use_curriculum_learning
        ).to(self.device)
        if self.args.is_enhanced:
            booster = STBoost(args=self.args, device=self.device, logger=self.logger).to(self.device)
            model = ParallelStructure(
                base_model=base_model,
                booster=booster,
                args=self.args,
                logger=self.logger,
                base_p_len=self.args.seq_len,
                base_q_len=self.args.horizon,
                base_model_name="MegaCRN"
            )
            return model.to(self.device)
        else:
            return base_model

    def _prepare_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
                  y shape (horizon, batch_size, num_sensor, input_dim)
        :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
                  y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        x0 = x[..., :self.args.input_dim]
        y0 = y[..., :self.args.output_dim]
        y1 = y[..., self.args.output_dim:]
        x0 = torch.from_numpy(x0).float()
        y0 = torch.from_numpy(y0).float()
        y1 = torch.from_numpy(y1).float()
        return x0.to(self.device), y0.to(self.device), y1.to(self.device)  # x, y, y_cov

    def _check_pretrained_model(self):
        return os.path.exists(self.pretrained_path)

    def _build_base_prediction_cache_for_loader(self, base_model, loader, mode):
        cache_batch_size = loader.batch_size
        base_p_len = self.args.seq_len
        base_q_len = self.args.horizon
        if self._load_stage2_inference_cache(mode, loader, base_p_len, base_q_len) is not None:
            return
        y_base_cache = None
        was_training = base_model.training
        base_model.eval()
        self.logger.info(
            f"Building MegaCRN base prediction cache for {mode}, batch_size={cache_batch_size}."
        )

        with torch.no_grad():
            for start in range(0, loader.total_samples, cache_batch_size):
                end = min(start + cache_batch_size, loader.total_samples)
                batch_indices = loader.indices[start:end]
                x, _y, ycov = self._prepare_x_y(loader.xs[batch_indices], loader.ys[batch_indices])
                x_base = x[:, -self.args.seq_len:]
                if self.args.stb_stage2_align_inputs:
                    x_base = convert_standardized_scale(x_base, self.scaler, self.base_scaler)
                ycov_base = ycov[:, : self.args.horizon]
                y_base, *_ = base_model(x_base, ycov_base)                         # (B, Q, S, 1)
                if self.args.stb_stage2_align_inputs:
                    y_base = convert_standardized_scale(y_base, self.base_scaler, self.scaler)
                y_base_np = y_base.detach().cpu().numpy().astype(np.float32)
                if y_base_cache is None:
                    y_base_cache = np.empty((loader.original_size,) + y_base_np.shape[1:], dtype=np.float32)
                y_base_cache[batch_indices] = y_base_np

        if y_base_cache.shape[0] != loader.original_size:
            raise RuntimeError(
                f"MegaCRN base cache size mismatch for {mode}: "
                f"{y_base_cache.shape[0]} vs {loader.original_size}."
            )
        loader.extra_inputs['y_base_cache'] = y_base_cache
        base_model.train(was_training)
        self.logger.info(f"Cached MegaCRN base predictions for {mode}: {y_base_cache.shape}.")
        self._save_stage2_inference_cache(mode, loader, y_base_cache, base_p_len, base_q_len)

    def _build_stage2_base_prediction_cache(self, model, modes=('train', 'val')):
        if not self.args.is_enhanced or not self.args.stb_use_base_prediction:
            return
        if not self.args.stb_cache_base_prediction:
            return

        cache_start_time = time.time()
        self.logger.info("=== Building frozen MegaCRN base prediction cache for STBoost stage2 ===")
        state_rng = save_rng_state(self.device)
        try:
            for mode in modes:
                self._build_base_prediction_cache_for_loader(
                    model.base_model,
                    self.data[f'{mode}_loader'],
                    mode
                )
        finally:
            restore_rng_state(state_rng, self.device)
        self.logger.info(f"MegaCRN base prediction cache inference time for {list(modes)}: {time.time() - cache_start_time:.2f}s.")

    def _clear_stage2_base_prediction_cache(self, modes):
        if not self.args.stb_cache_base_prediction:
            return

        released = False
        for mode in modes:
            loader = self.data.get(f'{mode}_loader')
            if loader is not None and loader.extra_inputs.pop('y_base_cache', None) is not None:
                released = True
                self.logger.info(f"Released cached MegaCRN base predictions for {mode}.")
        if released:
            gc.collect()

    def _get_optimizer(self, model, stage):
        if stage == 1:
            if self.args.is_enhanced:
                trainable_params = [
                    {"params": model.base_model.parameters()}
                ]
            else:
                trainable_params = [
                    {"params": model.parameters()}
                ]
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr, eps=self.args.epsilon)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.steps,
                                                                     gamma=self.args.lr_decay_ratio)
        else:
            trainable_params = [
                {"params": model.booster.parameters()},
                {"params": [model.weight]}
            ]
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.stb_base_lr, eps=1e-8)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=getattr(self.args, 'stb_lr_milestones', [50, 70, 90]),
                gamma=0.5
            )
        return optimizer, lr_scheduler

    def _compute_loss(self, y_pred, y_true, stage, query=None, pos=None, neg=None):
        assert stage == 1 or stage == 2, "stage must be 1 or 2, got stage: {}.".format(stage)
        if stage == 1:
            assert all(v is not None for v in [query, pos, neg]), "query and pos and neg should not be None"
            loss1 = masked_mae_loss(y_pred, y_true, use_adaptive_mask=False)     # masked_mae_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + self.args.lamb * loss2 + self.args.lamb1 * loss3
            return loss
        else:
            loss = masked_mae_loss(y_pred, y_true) + self.args.stb_lamb_mse * masked_mse_loss(y_pred, y_true)
            return loss

    def traintest_model(self):
        # main training process
        self.logger.info(self.args.dataset, 'training and testing started', time.ctime())

        model = self._get_model()

        if self.args.is_enhanced:
            self.logger.info("=== Start Enhanced ST Model Pipeline ===")
            # stage 1
            self._init_data(stage=1)
            self.logger.info("=== Stage 1: Training Base Model ===")
            model.set_stage(1)
            model.booster.eval()
            model.base_model.train()
            self._print_model(model)
            if self.args.stb_use_pretrained and self._check_pretrained_model():
                self.logger.info("loading pretrained base model")
                model = self._final_evaluation(model, stage=1)
            else:
                state_rng = save_rng_state(self.device)
                model = self._train_base_model(model)
                restore_rng_state(state_rng, self.device)          # recall the random state for reproducibility
                model = self._final_evaluation(model, stage=1)

            if self.args.stb_use_period_memorizer:
                # enhance general pattern
                align_ppm_scaler = bool(self.args.stb_ppm_align_scaler)
                ppm_rng_state = save_rng_state(self.device) if self.args.stb_ppm_control_rng else None
                ssl_dataloader = SelfSupervisedDataLoader(
                    train_data=self.train_data,
                    batch_size=64,
                    day_slots=(24 * 60) // self.args.duration,
                    holiday_path=self.args.traffic_df_filename[: self.args.traffic_df_filename.rfind('/')] + "/holidays.csv",
                    scaler=self.scaler,
                    num_pattern=self.args.stb_num_period_pattern,
                    pad_with_random_sample=bool(self.args.stb_ppm_pad_pairs),
                    shuffle=True,
                    reshuffle_each_epoch=bool(self.args.stb_ppm_reshuffle_each_epoch)
                )
                if align_ppm_scaler or ppm_rng_state is not None:
                    self.logger.info(
                        f"[PPM Pipeline] align_scaler={align_ppm_scaler}, "
                        f"pairs={ssl_dataloader.original_pair_count}, padding={ssl_dataloader.num_padding}, "
                        f"batches={ssl_dataloader.num_batch}, "
                        f"reshuffle_each_epoch={ssl_dataloader.reshuffle_each_epoch}"
                    )
                period_memorizer = PeriodPatternMemorizer(
                    num_pattern=self.args.stb_num_period_pattern,
                    num_sensors=self.args.num_nodes,
                    day_slots=(24 * 60) // self.args.duration,
                    period_mem_dim=self.args.stb_period_mem_dim,
                    init_values=ssl_dataloader.get_init_values()
                ).to(self.device)

                if self.args.stb_ppm_mode == 'full':
                    train_ssl_general_pattern(
                        ssl_dataloader,
                        period_memorizer,
                        self.logger,
                        self.device
                    )
                else:
                    self.logger.info("Use the historical mean initialization without PPM calibration.")
                period_memorizer.fix_pattern_memory(scaler=ssl_dataloader.scaler if align_ppm_scaler else None)
                model.set_period_pattern_memorizer(period_memorizer)
                if ppm_rng_state is not None:
                    restore_rng_state(ppm_rng_state, self.device)
                    self.logger.info('[PPM Pipeline] Restored the pre-PPM RNG state for Stage 2.')

                # debug ssl process
                ssl_dataloader.debug(self.save_path)
                period_memorizer.debug(self.save_path)
            else:
                self.logger.info("Skip Period Pattern Memorizer for this STBoost run.")

            # stage 2
            self._init_data(stage=2, categories=('train', 'val'))
            if self.args.stb_stage2_align_inputs:
                # Keep the same scaler bridge in cached and online frozen-base inference.
                model.base_scaler = self.base_scaler
                model.stage2_scaler = self.scaler
                self.logger.info('[PPM Pipeline] Stage 2 uses actual timestamps and aligned base-model input/output scales.')
            if self.args.stb_use_period_memorizer and self.args.stb_ppm_align_scaler:
                period_memorizer.set_retrieval_scaler(self.scaler)
                period_memorizer.debug(self.save_path)
            if self.args.stb_use_period_memorizer and (align_ppm_scaler or self.args.stb_ppm_control_rng):
                self.logger.info(
                    f"[PPM Pipeline] storage_scale={'raw' if align_ppm_scaler else 'PPM standardized'}, "
                    f"PPM scaler mean={ssl_dataloader.scaler.mean:.12g}, std={ssl_dataloader.scaler.std:.12g}; "
                    f"Stage 2 scaler mean={self.scaler.mean:.12g}, std={self.scaler.std:.12g}"
                )
            if self.args.stb_ppm_control_rng:
                indices_hash = hashlib.sha256(self.data['train_loader'].indices.tobytes()).hexdigest()
                self.logger.info(f"[PPM Pipeline] Stage 2 train indices SHA256: {indices_hash}")
            self.logger.info("=== Stage 2: Training STBoost Model ===")
            model.set_stage(2)
            model.booster.train()
            model.base_model.eval()
            self._build_stage2_base_prediction_cache(model)
            self._slim_stage2_value_channels(['train', 'val'])
            self._print_model(model)
            model = self._train_enhanced_model(model)
            self._clear_stage2_base_prediction_cache(['train', 'val'])
            self._release_data_loaders(['train', 'val'])
            self._init_data(stage=2, categories=('test',))
            self.profile_online_inference(model, stage=2, checkpoint_path=self.best_model_path)
            self._build_stage2_base_prediction_cache(model, modes=('test',))
            self._slim_stage2_value_channels(['test'])
            model = self._final_evaluation(model, stage=2)
            self._clear_stage2_base_prediction_cache(['test'])
            self._release_data_loaders(['test'])
            self.logger.info(self.args.dataset, 'training and testing ended', time.ctime())
        else:
            self.logger.info("=== Start Original ST Model Pipeline ===")
            # stage 1
            self._init_data(stage=1)
            self.logger.info("=== Stage 1: Training Base Model ===")
            model.train()
            self._print_model(model)
            model = self._train_base_model(model)
            model = self._final_evaluation(model, stage=1)
            self.logger.info(self.args.dataset, 'training and testing ended', time.ctime())
        return model

    def _train_base_model(self, model):
        optimizer, lr_scheduler = self._get_optimizer(model, 1)
        min_val_loss = float('inf')
        wait = 0
        abnormal_count = 0  # abnormal situation count (Gradient Vanishing)
        batches_seen = 0
        total_time_cost = 0
        gpu_memory_used = {}
        for epoch_num in range(self.args.epochs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()
            if self.args.is_enhanced:
                model.booster.eval()
                model.base_model.train()
            else:
                model.train()
            data_iter = self.data['train_loader'].get_iterator()
            losses = []
            for x, y, extra_inputs in data_iter:
                optimizer.zero_grad()
                x, y, ycov = self._prepare_x_y(x, y)
                x_missing_mask = extra_inputs.get('x_missing_mask')
                x_missing_mask = torch.from_numpy(x_missing_mask).bool().to(self.device) if x_missing_mask is not None else None

                if self.args.is_enhanced:
                    output, h_att, query, pos, neg = model(
                        x, ycov, y, batches_seen, x_missing_mask=x_missing_mask
                    )
                else:
                    output, h_att, query, pos, neg = model(x, ycov, y, batches_seen)

                y_true = self.scaler.inverse_transform(y)
                y_pred = self.scaler.inverse_transform(output)  # (B, Q, S, 1)
                loss = self._compute_loss(y_pred, y_true, 1, query=query, pos=pos, neg=neg)
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()

            train_loss = np.mean(losses)
            val_loss, _, _ = self._evaluate(model, 'val', stage=1)
            end_time = time.time()
            total_time_cost += (end_time - start_time)
            epoch_peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3  # Get max VRAM use in this epoch
            gpu_memory_used[epoch_num + 1] = epoch_peak_mem
            message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s, {:.2f}GB'.format(
                epoch_num + 1, self.args.epochs, batches_seen, train_loss, val_loss,
                optimizer.param_groups[0]['lr'], (end_time - start_time), epoch_peak_mem)
            self.logger.info(message)
            lr_scheduler.step()

            if math.isnan(train_loss) or math.isnan(val_loss):
                self.logger.info(f"NaN encountered: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                abnormal_count += 1
                if abnormal_count >= 5:  # Custom tolerance count
                    self.logger.info("NaN detected consecutively, terminating training early!")
                    break
            elif val_loss < min_val_loss:
                if val_loss <= 0.0 or train_loss <= 0.0:
                    self.logger.info(f"Abnormal MAE value: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                    abnormal_count += 1
                    if abnormal_count >= 5:  # Custom tolerance count
                        self.logger.info("Abnormal MAE detected consecutively, terminating training early!")
                        break
                else:
                    wait = 0
                    abnormal_count = 0  # reset counter
                    min_val_loss = val_loss
                    if self.args.is_enhanced:
                        torch.save(model.base_model.state_dict(), self.pretrained_path)
                    else:
                        torch.save(model.state_dict(), self.best_model_path)
            else:
                # 'elif val_loss >= min_val_loss:' can not deal with the nan case. Using 'else' instead can cope with all cases.
                if val_loss > 100.0:
                    self.logger.info(f"Abnormal MAE value: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                    abnormal_count += 1
                    if abnormal_count >= 5:  # Custom tolerance count
                        self.logger.info("Abnormal MAE detected consecutively, terminating training early!")
                        break
                else:
                    wait += 1
                    if wait >= self.args.patience:
                        self.logger.info('Early stopping at epoch: %d' % (epoch_num + 1))
                        break

        self.logger.info('The training time of per epoch is {:.4f} seconds!'.format(
            total_time_cost / (epoch_num + 1)
        ))
        self.logger.info('The total training time is {:.4f} seconds!'.format(total_time_cost))
        memory_report(gpu_memory_used, self.logger)
        return model

    def _train_enhanced_model(self, model):
        optimizer, lr_scheduler = self._get_optimizer(model, 2)
        min_val_loss = float('inf')
        wait = 0
        abnormal_count = 0
        unified_init_logit = float(getattr(self.args, 'stb_unified_init_logit', 4.0))
        unified_switch_epoch = max(1, int(getattr(self.args, 'stb_unified_switch_epoch', 80)))
        stb_batches_seen = 0
        total_time_cost = 0
        gpu_memory_used = {}
        for epoch in range(self.args.stb_epochs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()
            model.booster.train()
            model.base_model.eval()
            data_iter = self.data['train_loader'].get_iterator()
            if self.args.stb_use_curriculum_learning and getattr(self.args, "stb_step_size", None) is None:
                batch_per_epoch = self.data['train_loader'].num_batch
                total_steps = batch_per_epoch * self.args.stb_epochs
                curriculum_step_size = math.ceil(self.args.stb_reach_frac * total_steps / self.args.q_len)
                self.args.stb_step_size = curriculum_step_size
                self.logger.info(f"Curriculum Learning enabled. Step size set to {self.args.stb_step_size}.")
            losses = []
            if self.args.stb_is_unified_first:
                if not model.get_unified_training():
                    model.set_unified_training()
                    wait = 0
                    abnormal_count = 0
                    min_val_loss = float('inf')
                    with torch.no_grad():
                        model.weight.fill_(unified_init_logit)
                    self.logger.info(
                        f"Directly enter unified training! "
                        f"Reset val baseline and fusion weight to {unified_init_logit:.1f}."
                    )
            else:
                if epoch + 1 >= unified_switch_epoch and not model.get_unified_training():
                    model.set_unified_training()
                    wait = 0
                    abnormal_count = 0
                    min_val_loss = float('inf')
                    model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
                    with torch.no_grad():
                        model.weight.fill_(unified_init_logit)
                    self.logger.info(
                        f"Directly enter unified training, when epoch is {unified_switch_epoch}! "
                        f"Reset val baseline and fusion weight to {unified_init_logit:.1f}."
                    )
            for x, y, extra_inputs in data_iter:
                optimizer.zero_grad()
                timestamps = extra_inputs['timestamps']
                x_missing_mask = extra_inputs.get('x_missing_mask')
                x_missing_mask = torch.from_numpy(x_missing_mask).bool().to(self.device) if x_missing_mask is not None else None
                y_base_cache_np = extra_inputs.get('y_base_cache')
                y_base_cache = (
                    torch.from_numpy(y_base_cache_np).float().to(self.device)
                    if y_base_cache_np is not None else None
                )
                self.time_interpreter.set_timestamps(timestamps)
                time_encoding = self.time_interpreter.get_time_encoding()
                time_info = self.time_interpreter.get_time_info(self.args.stb_num_period_pattern)
                x, y, ycov = self._prepare_x_y(x, y)

                output = model(
                    x, ycov, time_encoding, time_info, self.boost_dist_adj, stb_batches_seen,
                    x_missing_mask=x_missing_mask, y_base_cache=y_base_cache
                )

                y_pred = self.scaler.inverse_transform(output)                      # (B, Q, S, 1)
                y_true = self.scaler.inverse_transform(y[:, : output.shape[1]])     # curriculum learning
                loss = self._compute_loss(y_pred, y_true, 2)
                losses.append(loss.item())
                stb_batches_seen += 1
                loss.backward()
                optimizer.step()
                self.time_interpreter.reset_timestamps()

            train_loss = np.mean(losses)
            val_loss, _, _ = self._evaluate(model, 'val', stage=2)
            end_time = time.time()
            total_time_cost += (end_time - start_time)
            epoch_peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3  # Get max VRAM use in this epoch
            gpu_memory_used[epoch + 1] = epoch_peak_mem
            message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s, {:.2f}GB'.format(
                epoch + 1, self.args.stb_epochs, stb_batches_seen, train_loss, val_loss,
                optimizer.param_groups[0]['lr'], (end_time - start_time), epoch_peak_mem)
            self.logger.info(message)
            lr_scheduler.step()

            if math.isnan(train_loss) or math.isnan(val_loss):
                self.logger.info(f"NaN encountered: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                abnormal_count += 1
                if abnormal_count >= 5:  # Custom tolerance count
                    self.logger.info("NaN detected consecutively, terminating training early!")
                    break
            elif val_loss < min_val_loss:
                if val_loss <= 0.0 or train_loss <= 0.0:
                    self.logger.info(
                        f"Abnormal MAE value: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                    abnormal_count += 1
                    if abnormal_count >= 5:  # Custom tolerance count
                        self.logger.info("Abnormal MAE detected consecutively, terminating training early!")
                        break
                else:
                    wait = 0
                    abnormal_count = 0  # reset counter
                    self.logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, self.best_model_path))
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), self.best_model_path)
            else:
                # 'elif val_loss >= min_val_loss:' can not deal with the nan case. Using 'else' instead can cope with all cases.
                if val_loss > 300.0:
                    self.logger.info(f"Abnormal MAE value: train_mae={train_loss}, val_mae={val_loss}, ignoring this update!")
                    abnormal_count += 1
                    if abnormal_count >= 5:  # Custom tolerance count
                        self.logger.info("Abnormal MAE detected consecutively, terminating training early!")
                        break
                else:
                    wait += 1
                    if wait >= self.args.stb_patience:
                        if not self.args.stb_is_unified_first:
                            if not model.get_unified_training():
                                model.set_unified_training()
                                wait = 0
                                abnormal_count = 0
                                min_val_loss = float('inf')
                                model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
                                with torch.no_grad():
                                    model.weight.fill_(unified_init_logit)
                                self.logger.info(
                                    f"Enter unified training! "
                                    f"Reset val baseline and fusion weight to {unified_init_logit:.1f}."
                                )
                            else:
                                self.logger.info('Early stopping at epoch: %d' % (epoch + 1))
                                break
                        else:
                            self.logger.info('Early stopping at epoch: %d' % (epoch + 1))
                            break

        self.logger.info('The training time of per epoch is {:.4f} seconds!'.format(
            total_time_cost / (epoch + 1)
        ))
        self.logger.info('The total training time is {:.4f} seconds!'.format(total_time_cost))
        memory_report(gpu_memory_used, self.logger)
        return model

    @staticmethod
    def _init_stage2_debug_records():
        return {
            'x_input': [],
            'x_temporal': [],
            'x_period': [],
            'x_residual': [],
            'y_true': [],
            'y_base': [],
            'y_temporal': [],
            'y_period': [],
            'y_residual': [],
            'y_boost': [],
            'y_out': [],
            'x_missing_mask': [],
            'time_series_emb': [],
            'hidden': [],
            'time_encoding': [],
            'time_info': [],
            'timestamps': []
        }

    def _append_stage2_debug_records(
            self,
            debug_records,
            x,
            y_true,
            y_base,
            y_boost,
            y_out,
            x_missing_mask,
            time_encoding,
            time_info,
            timestamps,
            booster_debug
    ):
        scaler_std = float(self.scaler.std)

        debug_records['x_input'].append(
            self.scaler.inverse_transform(x).cpu().numpy()
        )
        debug_records['x_temporal'].append(
            self.scaler.inverse_transform(booster_debug['x_temporal']).cpu().numpy()
        )
        debug_records['x_period'].append(
            self.scaler.inverse_transform(booster_debug['x_period']).cpu().numpy()
        )
        debug_records['x_residual'].append(
            (booster_debug['x_residual'] * scaler_std).cpu().numpy()
        )
        debug_records['y_true'].append(y_true[:, : y_out.shape[1]].cpu().numpy())
        debug_records['y_base'].append(
            self.scaler.inverse_transform(y_base).cpu().numpy()
        )
        debug_records['y_temporal'].append(
            self.scaler.inverse_transform(booster_debug['y_temporal']).cpu().numpy()
        )
        debug_records['y_period'].append(
            self.scaler.inverse_transform(booster_debug['y_period']).cpu().numpy()
        )
        debug_records['y_residual'].append(
            (booster_debug['y_residual'] * scaler_std).cpu().numpy()
        )
        debug_records['y_boost'].append(
            self.scaler.inverse_transform(y_boost).cpu().numpy()
        )
        debug_records['y_out'].append(
            self.scaler.inverse_transform(y_out).cpu().numpy()
        )
        debug_records['x_missing_mask'].append(
            booster_debug['x_missing_mask'].cpu().numpy()
        )
        debug_records['time_series_emb'].append(
            booster_debug['time_series_emb'].cpu().numpy()
        )
        debug_records['hidden'].append(
            booster_debug['hidden'].cpu().numpy()
        )
        debug_records['time_encoding'].append(time_encoding.cpu().numpy())
        debug_records['time_info'].append(np.asarray(time_info))
        debug_records['timestamps'].append(np.asarray(timestamps))

    def _save_stage2_debug_records(self, debug_records, mode):
        debug_dir = os.path.join(self.save_path, 'debug_arrays', mode)
        os.makedirs(debug_dir, exist_ok=True)
        for key, values in debug_records.items():
            concat_value = np.concatenate(values, axis=0)
            np.save(os.path.join(debug_dir, f'{key}.npy'), concat_value)
        self.logger.info(f'Saved {mode} debug arrays to {debug_dir}')

    def _evaluate(self, model, mode, stage, dump_debug=False, store_on_cpu=False):
        with torch.no_grad():
            model.eval()
            data_iter = self.data[f'{mode}_loader'].get_iterator()
            losses = []
            ys_true, ys_pred = [], []
            debug_records = None
            if stage == 2 and self.args.is_enhanced and dump_debug:
                debug_records = self._init_stage2_debug_records()
            for x, y, extra_inputs in data_iter:
                x, y, ycov = self._prepare_x_y(x, y)
                timestamps = extra_inputs['timestamps']
                x_missing_mask = extra_inputs.get('x_missing_mask')
                x_missing_mask = torch.from_numpy(x_missing_mask).bool().to(self.device) if x_missing_mask is not None else None

                if stage == 1:
                    if self.args.is_enhanced:
                        output, h_att, query, pos, neg = model(x, ycov, x_missing_mask=x_missing_mask)
                    else:
                        output, h_att, query, pos, neg = model(x, ycov)
                    y_pred = self.scaler.inverse_transform(output)
                    if mode != 'test':
                        y_true = self.scaler.inverse_transform(y)
                    else:
                        y_true = y
                    loss = self._compute_loss(y_pred, y_true, 1, query=query, pos=pos, neg=neg)
                elif stage == 2 and self.args.is_enhanced:
                    self.time_interpreter.set_timestamps(timestamps)
                    time_encoding = self.time_interpreter.get_time_encoding()
                    time_info = self.time_interpreter.get_time_info(self.args.stb_num_period_pattern)
                    y_base_cache_np = extra_inputs.get('y_base_cache')
                    y_base_cache = (
                        torch.from_numpy(y_base_cache_np).float().to(self.device)
                        if y_base_cache_np is not None else None
                    )
                    output = model(
                        x, ycov, time_encoding, time_info, self.boost_dist_adj,
                        x_missing_mask=x_missing_mask, y_base_cache=y_base_cache
                    )
                    y_pred = self.scaler.inverse_transform(output)
                    if mode != 'test':
                        y_true = self.scaler.inverse_transform(y)
                    else:
                        y_true = y
                    if debug_records is not None:
                        stage2_debug = model.latest_stage2_debug
                        assert stage2_debug is not None, "model.latest_stage2_debug should be populated in debug mode."
                        booster_debug = stage2_debug['booster_debug']
                        self._append_stage2_debug_records(
                            debug_records=debug_records,
                            x=x,
                            y_true=y_true,
                            y_base=stage2_debug['y_base'],
                            y_boost=stage2_debug['y_boost'],
                            y_out=stage2_debug['y_out'],
                            x_missing_mask=x_missing_mask,
                            time_encoding=time_encoding,
                            time_info=time_info,
                            timestamps=timestamps,
                            booster_debug=booster_debug
                        )
                    loss = self._compute_loss(y_pred, y_true, 2)
                else:
                    raise NotImplementedError("got stage: {}, is_enhanced: {}".format(
                        stage, self.args.is_enhanced
                    ))

                losses.append(loss.item())
                # Important for MegaCRN model to let T come first.
                y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
                ys_true.append(y_true.detach().cpu() if store_on_cpu else y_true)
                ys_pred.append(y_pred.detach().cpu() if store_on_cpu else y_pred)
            mean_loss = np.mean(losses)
            # (horizon, n_len, n_sensors, 1)
            if debug_records is not None:
                self._save_stage2_debug_records(debug_records, mode)
            return mean_loss, torch.concat(ys_true, dim=1).cpu().numpy(), torch.concat(ys_pred, dim=1).cpu().numpy()

    def _final_evaluation(self, model, stage):
        assert stage == 1 or stage == 2, "stage must be 1 or 2, got stage: {}.".format(stage)
        self.logger.info('=' * 35 + 'Best model performance' + '=' * 35)
        if self.args.is_enhanced:
            if stage == 1:
                model.base_model.load_state_dict(torch.load(self.pretrained_path, map_location=self.device, weights_only=True))
            else:
                model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        else:
            model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        if not self.args.is_enhanced:
            self.profile_online_inference(model, stage=1, checkpoint_path=self.best_model_path)
        test_start_time = time.time()
        test_loss, y_true, y_pred = self._evaluate(model, 'test', stage)
        test_end_time = time.time()
        self.logger.info(f'The total test time is {test_end_time - test_start_time}s!')

        evaluate_all_metrics(
            np.transpose(y_pred[..., 0], (1, 2, 0)),
            np.transpose(y_true[..., 0], (1, 2, 0)),
            0.,
            self.logger,
            use_adaptive_mask=(stage == 2)
        )
        if stage == 2 and self.args.is_enhanced and self.args.stb_dump_debug_arrays:
            self.logger.info('==== Dumping full stage-2 debug arrays for train / val / test ====')
            for mode in ['train', 'val', 'test']:
                self._evaluate(model, mode, stage, dump_debug=True)
        return model

    def get_pretrained_enhanced_model(self):
        return self._load_pretrained_enhanced_model()

    def _print_MegaCRN_args(self):
        self.logger.info('========== MegaCRN args: ==========')
        self.logger.info('base_seed:', self.base_seed)
        self.logger.info('trainval_ratio:', self.args.trainval_ratio)
        self.logger.info('val_ratio:', self.args.val_ratio)
        self.logger.info('seq_len', self.args.seq_len)
        self.logger.info('horizon:', self.args.horizon)
        self.logger.info('input_dim:', self.args.input_dim)
        self.logger.info('output_dim:', self.args.output_dim)
        self.logger.info('num_rnn_layers:', self.args.num_rnn_layers)
        self.logger.info('rnn_units:', self.args.rnn_units)
        self.logger.info('max_diffusion_step:', self.args.max_diffusion_step)
        self.logger.info('mem_num:', self.args.mem_num)
        self.logger.info('mem_dim:', self.args.mem_dim)
        self.logger.info('loss:', self.args.loss)
        self.logger.info('separate loss lamb:', self.args.lamb)
        self.logger.info('compact loss lamb1:', self.args.lamb1)
        self.logger.info('base_batch_size:', self.args.base_batch_size)
        self.logger.info('epochs:', self.args.epochs)
        self.logger.info('patience:', self.args.patience)
        self.logger.info('lr:', self.args.lr)
        self.logger.info('epsilon:', self.args.epsilon)
        self.logger.info('steps:', self.args.steps)
        self.logger.info('lr_decay_ratio:', self.args.lr_decay_ratio)
        self.logger.info('use_curriculum_learning:', self.args.use_curriculum_learning)
