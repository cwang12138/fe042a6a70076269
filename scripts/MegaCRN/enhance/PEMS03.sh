#!/bin/bash

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../.."

base_model='MegaCRN'
is_enhanced=1
dataset='PEMS03'
p_len=96
gpu=${GPU:-0}

for q_len in ${STB_Q_LIST:-12 24 48 96}; do
  "${PYTHON_BIN:-python}" -u main.py \
    --base_model $base_model \
    --is_enhanced $is_enhanced \
    --dataset $dataset \
    --start_date_str '2018-09-01' \
    --end_date_str '2018-11-30' \
    --seq_len 12 \
    --horizon 12 \
    --num_nodes 358 \
    --traffic_df_filename './datasets/PEMS03/PEMS03.h5' \
    --trainval_ratio 0.8 \
    --val_ratio 0.125 \
    --input_dim 1 \
    --output_dim 1 \
    --max_diffusion_step 3 \
    --num_rnn_layers 1 \
    --rnn_units 64 \
    --mem_num 20 \
    --mem_dim 64 \
    --loss 'mask_mae_loss' \
    --lamb 0.01 \
    --lamb1 0.01 \
    --epochs 200 \
    --patience 20 \
    --base_batch_size 64 \
    --base_seed 100 \
    --stb_init_seed ${STB_INIT_SEED:-2025} \
    --lr 0.01 \
    --steps '[50, 100]' \
    --lr_decay_ratio 0.1 \
    --epsilon '1e-3' \
    --max_grad_norm 5 \
    --use_curriculum_learning \
    --cl_decay_steps 2000 \
    --test_every_n_epochs 5 \
    --gpu $gpu \
    --p_len $p_len \
    --q_len $q_len \
    --stb_exp_category "${STB_EXP_CATEGORY:-}" \
    --stb_exp_name "${STB_EXP_NAME:-}" \
    --stb_dynamic_spatial_embed_dim ${STB_DYNAMIC_SPATIAL_EMBED_DIM:-16} \
    --stb_num_hyper_nodes ${STB_NUM_HYPER_NODES:-16} \
    --stb_dynamic_cheb_k ${STB_DYNAMIC_CHEB_K:-1} \
    --stb_time_embed_dim 4 \
    --stb_num_period_pattern 2 \
    --stb_period_mem_dim 1 \
    --stb_static_spatial_embed_dim 8 \
    --stb_static_cheb_k 3 \
    --stb_static_hid_dim 32 \
    --stb_temporal_embed_dim ${STB_TEMPORAL_EMBED_DIM:-32} \
    --stb_st_mixer_depth ${STB_ST_MIXER_DEPTH:-3} \
    --stb_temporal_dropout 0.15 \
    --stb_use_input_temporal ${STB_USE_INPUT_TEMPORAL:-1} \
    --stb_use_input_period ${STB_USE_INPUT_PERIOD:-1} \
    --stb_use_input_residual ${STB_USE_INPUT_RESIDUAL:-0} \
    --stb_use_period_memorizer ${STB_USE_PERIOD_MEMORIZER:-1} \
    --stb_ppm_align_scaler ${STB_PPM_ALIGN_SCALER:-1} \
    --stb_ppm_pad_pairs ${STB_PPM_PAD_PAIRS:-0} \
    --stb_ppm_reshuffle_each_epoch ${STB_PPM_RESHUFFLE_EACH_EPOCH:-1} \
    --stb_ppm_control_rng ${STB_PPM_CONTROL_RNG:-1} \
    --stb_stage2_align_inputs ${STB_STAGE2_ALIGN_INPUTS:-1} \
    --stb_ppm_mode ${STB_PPM_MODE:-full} \
    --stb_sidecar_mode ${STB_SIDECAR_MODE:-full} \
    --stb_mlp_hidden_dim ${STB_MLP_HIDDEN_DIM:-0} \
    --stb_use_sample_wise_adaptive_graph ${STB_USE_SAMPLE_WISE_ADAPTIVE_GRAPH:-1} \
    --stb_use_static_graph ${STB_USE_STATIC_GRAPH:-1} \
    --stb_use_spatial_modeling ${STB_USE_SPATIAL_MODELING:-1} \
    --stb_use_base_prediction ${STB_USE_BASE_PREDICTION:-1} \
    --stb_cache_base_prediction ${STB_CACHE_BASE_PREDICTION:-1} \
    --stb_use_inference_cache ${STB_USE_INFERENCE_CACHE:-0} \
    --stb_inference_cache_root "${STB_INFERENCE_CACHE_ROOT:-./experiments/inference_cache}" \
    --stb_batch_size ${STB_BATCH_SIZE:-64} \
    --stb_use_tod ${STB_USE_TOD:-1} \
    --stb_use_dow ${STB_USE_DOW:-1} \
    --stb_use_pretrained ${STB_USE_PRETRAINED:-1} \
    --stb_epochs 100 \
    --stb_base_lr 0.001 \
    --stb_patience 10 \
    --stb_unified_switch_epoch 60 \
    --stb_lr_milestones '[40, 70, 90]' \
    --stb_reach_frac 0.4
done
