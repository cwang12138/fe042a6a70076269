"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 13:47
# @Author  : Chen Wang
# @Site    : 
# @File    : args.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import argparse
import sys

from utils.checkpoint_config import load_checkpoint_arguments


class Args:
    def __init__(self, argv=None):
        argv = load_checkpoint_arguments(sys.argv[1:] if argv is None else argv)
        init_parser = argparse.ArgumentParser()
        init_parser.add_argument('--base_model', type=str, required=True, choices=['MegaCRN'], help="Specify the base model")
        init_parser.add_argument('--is_enhanced', type=int, required=True, help="")
        init_args, self.remaining_argv = init_parser.parse_known_args(argv)
        self.base_model = init_args.base_model
        self.is_enhanced = bool(init_args.is_enhanced)

        self.parser = argparse.ArgumentParser()
        self._create_args()

    def _create_args(self):
        self._init_shared_args()
        if self.base_model == 'MegaCRN':
            self._init_MegaCRN_args()
        else:
            raise ValueError(f'Please enter the vaild base model name! (input: {self.base_model})')
        if self.is_enhanced:
            self._init_STBoost_args()

    def get_args(self):
        args = self.parser.parse_args(self.remaining_argv)
        if args.online_inference_batch_size <= 0:
            self.parser.error('--online_inference_batch_size must be positive')
        if args.online_inference_only and (not args.best_model_path or not args.profile_online_inference):
            self.parser.error('--test_only/--online_inference_only requires --best_model_path and --profile_online_inference 1')
        if args.check_only and not args.online_inference_only:
            self.parser.error('--check_only requires --test_only 1')
        if args.online_inference_only and args.is_enhanced:
            # The full checkpoint already contains the backbone and PPM buffers.
            args.stb_use_inference_cache = 0
            args.stb_cache_base_prediction = 0
            args.stb_use_pretrained = 0
        return args

    def _init_shared_args(self):
        self.parser.add_argument(
            '--stb_init_seed', type=int, default=2025,
            help='seed for STBoost module initialization and fallback for base models without a dedicated seed; '
                 'the previous RNG state is restored after STBoost initialization'
        )
        self.parser.add_argument('--dataset', type=str,
                                 choices=['PEMS03', 'PEMS08'],
                                 default='PEMS03', help='which dataset to run')
        self.parser.add_argument('--duration', type=int, default=5,
                                 help='the time interval (minutes) between two time steps of the dataset')
        self.parser.add_argument('--num_nodes', type=int, default=358, help='num_nodes')
        self.parser.add_argument('--p_len', type=int, default=96, help='the length of long-term traffic prediction input sequence')
        self.parser.add_argument('--q_len', type=int, default=12, help='the length of long-term traffic prediction output sequence')
        self.parser.add_argument('--base_model', type=str, default=self.base_model, help='the name of base model')
        self.parser.add_argument('--is_enhanced', type=bool,
                                 default=self.is_enhanced,
                                 help='whether to use STBoost.')
        self.parser.add_argument('--best_model_path', type=str, default=None, help='the checkpoint of best model.')
        self.parser.add_argument('--checkpoint_config', type=str, default=None,
                                 help='test-only configuration; defaults to config.json beside best_model_path if present; '
                                      'explicit CLI arguments override saved settings')
        self.parser.add_argument('--check_only', action='store_true',
                                 help='validate test-only files/configuration/imports without constructing a trainer')
        self.parser.add_argument('--best_model_timestamps', type=str, default=None,
                                 help='the timestamp record of best model, e.g. 2025-06-04-22-55-32')
        self.parser.add_argument('--profile_online_inference', type=int, default=1, choices=[0, 1],
                                 help='measure uncached test-set forward time separately from metric evaluation')
        self.parser.add_argument('--online_inference_batch_size', type=int, default=16,
                                 help='shared Direct/STBoost inference batch size; does not change training batches')
        self.parser.add_argument('--test_only', '--online_inference_only', dest='online_inference_only',
                                 type=int, default=0, choices=[0, 1],
                                 help='load best_model_path, report test metrics and three inference timings without training')
        self.parser.add_argument('--start_date_str', type=str, default=None,
                                 help='the beginning date of dataset, e.g. 2023-01-01')
        self.parser.add_argument('--end_date_str', type=str, default=None,
                                 help='the end date of dataset, e.g. 2023-12-31')

    def _init_MegaCRN_args(self):
        self.parser.add_argument("--traffic_df_filename", type=str, default="datasets/PEMS03/PEMS03.h5", help="Raw traffic readings.")
        self.parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
        self.parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
        self.parser.add_argument('--trainval_ratio', type=float, default=0.8,
                                 help='the ratio of training and validation data among the total')
        self.parser.add_argument('--val_ratio', type=float, default=0.125,
                                 help='the ratio of validation data among the trainval ratio')
        self.parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
        self.parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
        self.parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
        self.parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
        self.parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
        self.parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
        self.parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
        self.parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
        self.parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
        self.parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
        self.parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
        self.parser.add_argument("--base_batch_size", type=int, default=64,
                                 help="batch size for the MegaCRN base model")
        self.parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
        self.parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
        self.parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
        self.parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
        self.parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
        self.parser.add_argument("--use_curriculum_learning", action="store_true", default=False,
                                 help="use_curriculum_learning")
        self.parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
        self.parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
        self.parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
        self.parser.add_argument(
            '--base_seed', type=int, default=100,
            help='seed for MegaCRN initialization, data ordering, and base-model training'
        )


    def _init_STBoost_args(self):
        # Model Settings
        self.parser.add_argument('--stb_dynamic_spatial_embed_dim', type=int, default=16,
                                 help='the embedding dim of the dynamic spatial features')
        self.parser.add_argument('--stb_num_hyper_nodes', type=int, default=16,
                                 help='the number of hyper-node memory representations used to condition the adaptive graph')
        self.parser.add_argument('--stb_dynamic_cheb_k', type=int, default=3, help='the k of DGC')
        self.parser.add_argument('--stb_time_embed_dim', type=int, default=4,
                                 help='time context embedding dim (fixed)')
        self.parser.add_argument('--stb_num_period_pattern', type=int, default=2,
                                 help='the number of period pattern the memory states retained (fixed)')
        self.parser.add_argument('--stb_period_mem_dim', type=int, default=1,
                                 help='the dimension of period pattern in STBoost')
        self.parser.add_argument('--stb_static_spatial_embed_dim', type=int, default=16, help='the embedding dim of the static spatial features')
        self.parser.add_argument('--stb_static_cheb_k', type=int, default=3, help='the k of GCN')
        self.parser.add_argument('--stb_static_hid_dim', type=int, default=32, help='')
        self.parser.add_argument('--stb_temporal_embed_dim', type=int, default=32,
                                 help='hidden dimension of the single-path temporal mixer')
        self.parser.add_argument('--stb_st_mixer_depth', type=int, default=3,
                                 help='number of repeated (DGC -> MLP) blocks after the input MLP')
        self.parser.add_argument('--stb_temporal_dropout', type=float, default=0.15,
                                 help='dropout of the single-path temporal mixer')
        self.parser.add_argument('--stb_use_input_temporal', type=int, default=1, choices=[0, 1],
                                 help='whether to use temporal observations in the STBoost input decomposition')
        self.parser.add_argument('--stb_use_input_period', type=int, default=1, choices=[0, 1],
                                 help='whether to use periodic prior in the STBoost input decomposition')
        self.parser.add_argument('--stb_use_input_residual', type=int, default=0, choices=[0, 1],
                                 help='whether to use residual signal in the STBoost input decomposition')
        self.parser.add_argument('--stb_use_period_memorizer', type=int, default=1, choices=[0, 1],
                                 help='whether to use the periodic pattern memorizer and its period prior')
        self.parser.add_argument('--stb_ppm_align_scaler', type=int, default=1, choices=[0, 1],
                                 help='store raw PPM prototypes and normalize retrieval with the Stage 2 scaler')
        self.parser.add_argument('--stb_ppm_pad_pairs', type=int, default=0, choices=[0, 1],
                                 help='repeat random pairs to fill the final PPM batch')
        self.parser.add_argument('--stb_ppm_reshuffle_each_epoch', type=int, default=1, choices=[0, 1],
                                 help='reshuffle PPM pairs before each calibration epoch')
        self.parser.add_argument('--stb_ppm_control_rng', type=int, default=1, choices=[0, 1],
                                 help='isolate PPM preprocessing and calibration RNG consumption from Stage 2')
        self.parser.add_argument('--stb_stage2_align_inputs', type=int, default=1, choices=[0, 1],
                                 help='use actual window timestamps and bridge base-model scaler coordinates in Stage 2')
        self.parser.add_argument('--stb_ppm_mode', type=str, default='full', choices=['full', 'mean_only'],
                                 help='whether to calibrate PPM or use its historical mean initialization directly')
        self.parser.add_argument('--stb_sidecar_mode', type=str, default='full', choices=['full', 'mlp'],
                                 help='whether to use the full STBoost sidecar or a parameter-matched MLP')
        self.parser.add_argument('--stb_mlp_hidden_dim', type=int, default=0,
                                 help='hidden dimension of the parameter-matched MLP sidecar')
        self.parser.add_argument('--stb_use_sample_wise_adaptive_graph', type=int, default=1, choices=[0, 1],
                                 help='whether to use Hyper-Node Memory conditioned sample-wise adaptive graph')
        self.parser.add_argument('--stb_use_static_graph', type=int, default=1, choices=[0, 1],
                                 help='whether to use static topology support in the graph branch')
        self.parser.add_argument('--stb_use_spatial_modeling', type=int, default=1, choices=[0, 1],
                                 help='whether to use spatial graph modeling; disabled means pure temporal modeling')
        self.parser.add_argument('--stb_use_base_prediction', type=int, default=1, choices=[0, 1],
                                 help='whether to use the short-term base model prediction in STBoost stage2')
        self.parser.add_argument('--stb_use_tod', type=int, default=1, choices=[0, 1],
                                 help='whether to concatenate time-of-day sin/cos features into the main input')
        self.parser.add_argument('--stb_use_dow', type=int, default=1, choices=[0, 1],
                                 help='whether to concatenate day-of-week sin/cos features into the main input')
        self.parser.add_argument('--stb_cache_base_prediction', type=int, default=1, choices=[0, 1],
                                 help='whether to cache frozen base-model predictions before STBoost stage2')
        self.parser.add_argument('--stb_use_inference_cache', type=int, default=0, choices=[0, 1],
                                 help='whether to persist and reuse frozen base-model predictions on disk')
        self.parser.add_argument('--stb_inference_cache_root', type=str, default='./experiments/inference_cache',
                                 help='root directory for persistent base prediction cache')
        self.parser.add_argument('--stb_batch_size', type=int, default=None,
                                 help='batch size for STBoost stage2, None means use base_batch_size')
        self.parser.add_argument('--stb_exp_category', type=str, default='',
                                 help='internal save-path category for STBoost experiment outputs')
        self.parser.add_argument('--stb_exp_name', type=str, default='',
                                 help='internal save-path name for STBoost experiment outputs')

        # Training Settings
        self.parser.add_argument('--stb_use_pretrained', type=int, nargs='?', const=1, default=0, choices=[0, 1],
                                 help='whether to load an existing pretrained base model when available')
        self.parser.add_argument('--stb_is_unified_first', action="store_true", default=False,
                                 help="whether to train STBoost and W_out in a unified manner first.")
        self.parser.add_argument('--stb_unified_switch_epoch', type=int, default=80,
                                 help='force switch to unified training at this epoch when not unified first')
        self.parser.add_argument('--stb_epochs', type=int, default=100, help='')
        self.parser.add_argument('--stb_base_lr', type=float, default=0.001, help='')
        self.parser.add_argument('--stb_lr_milestones', type=eval, default=[50, 70, 90],
                                 help='stage2 lr scheduler milestones for STBoost')
        self.parser.add_argument('--stb_patience', type=int, default=20, help='')
        self.parser.add_argument('--stb_use_curriculum_learning', action='store_true', default=False)
        self.parser.add_argument('--stb_step_size', type=int, default=None,
                                 help='manual curriculum step size; if None, auto-compute from stb_reach_frac')
        self.parser.add_argument('--stb_reach_frac', type=float, default=0.4,
                                 help='total curriculum exposure epoch ratio')
        self.parser.add_argument('--stb_unified_init_logit', type=float, default=4.0,
                                 help='initial logit of fusion weight in unified training')
        self.parser.add_argument('--stb_lamb_mse', type=float, default=0., help='')
        self.parser.add_argument('--stb_dump_debug_arrays', action='store_true', default=False,
                                 help='dump stage2 MegaCRN test-time debug arrays to npy files')
