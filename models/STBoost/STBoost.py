"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 21:36
# @Author  : Chen Wang
# @Site    : 
# @File    : STBoost.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from utils.tools import convert_standardized_scale


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_nodes, embed_dim, k=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.k = k

        self.init_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))      # (num_nodes, hidden_dim)

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        for _ in range(k - 2):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.gcn_layers.append(nn.Linear(hidden_dim, embed_dim))

        self.dropout = nn.Dropout(dropout)

    def normalize_adj(self, adj):
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        degree = adj.sum(dim=1).clamp(min=1).sqrt()
        norm_adj = adj / degree.view(-1, 1) / degree.view(1, -1)    # D^{-1/2} A D^{-1/2}
        return norm_adj

    def forward(self, dist_mx):
        """
        :param dist_mx: (S, S)
        :return: static node embedding: (S, D_spatial)
        """
        norm_adj = self.normalize_adj(dist_mx)

        # init
        node_embedding = self.init_embedding

        # aggregation
        for i in range(self.k):
            res = node_embedding
            node_embedding = torch.matmul(norm_adj, node_embedding)               # propagate information
            node_embedding = self.gcn_layers[i](node_embedding)
            if i < self.k - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = self.dropout(node_embedding)
                node_embedding = node_embedding + res                             # skip-connection

        return node_embedding           # (num_nodes, embed_dim)


class HyperNodeMemory(nn.Module):
    def __init__(self, input_dim, embed_dim, num_hyper_nodes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyper_nodes = num_hyper_nodes
        self.representations = nn.Parameter(torch.randn(num_hyper_nodes, embed_dim))
        self.query_proj = nn.Linear(input_dim, embed_dim, bias=True)

    def forward(self, x):
        """
        :param x: (B, S, D_in)
        :return:
            node_embeddings: (B, S, D_embed)
            attention_scores: (B, S, M)
        """
        # Project the node summaries into the hyper-node query space first.
        query = self.query_proj(x)                                             # (B, S, D_embed)
        # Match each node summary against the shared hyper-node representations.
        logits = torch.einsum('b s d, m d -> b s m', query, self.representations)         # (B, S, M)
        logits = logits / math.sqrt(self.embed_dim)
        attention_scores = torch.softmax(logits, dim=-1)                       # (B, S, M)
        # Recombine the hyper-node representations into sample-wise node embeddings.
        node_embeddings = torch.einsum('b s m, m d -> b s d', attention_scores, self.representations)
        return node_embeddings, attention_scores


class DynamicGraphConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, spatial_embed_dim, cheb_k):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(spatial_embed_dim, cheb_k, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(spatial_embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
        self.epsilon = nn.Parameter(torch.zeros(1))
        # Start from a topology-dominant fusion, then let the model relax if needed.
        self.static_mix_logit = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, node_embeddings, static_prior=None,
                use_sample_wise_adaptive_graph=True):
        """
        Propagate temporal features with either the final sample-wise graph or a static support.
        :param x: shape, (B, L, S, input_dim)
        :param node_embeddings:
            - sample-wise adaptive mode: (B, S, D)
            - static mode: (S, D)
        :param static_prior: shape, (S, S), static topology prior
        :return: shape, (B, L, S, output_dim)
        """
        if use_sample_wise_adaptive_graph:
            if node_embeddings.dim() != 3:
                raise ValueError("Sample-wise adaptive graph expects node embeddings with shape (B, S, D).")

            adaptive_logits = torch.einsum('b s d, b m d -> b s m', node_embeddings, node_embeddings)
            adaptive_logits = adaptive_logits * torch.exp(self.epsilon) / math.sqrt(node_embeddings.shape[-1])

            if static_prior is not None:
                static_support = static_prior / static_prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (S, S)
                static_support = static_support.unsqueeze(0)                                               # (1, S, S)

                candidate_mask = static_support > 0                                                        # (1, S, S)
                candidate_mask = candidate_mask | torch.eye(static_support.shape[-1], device=x.device,
                                                            dtype=torch.bool).unsqueeze(0)                # (1, S, S)

                # Build the sample-wise adaptive graph only on topology-allowed edges first.
                adaptive_logits = F.relu(adaptive_logits).masked_fill(~candidate_mask, float('-inf'))     # (B, S, S)
                adaptive_support = torch.softmax(adaptive_logits, dim=-1)                                  # (B, S, S)
                mix_ratio = torch.sigmoid(self.static_mix_logit)
                g = mix_ratio * static_support + (1 - mix_ratio) * adaptive_support                       # (B, S, S)
                g = g / g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                g = torch.softmax(F.relu(adaptive_logits), dim=-1)                                        # (B, S, S)

            # Generate sample-specific Chebyshev kernels directly from the node embeddings.
            weights = torch.einsum('b s d, d k i o -> b s k i o', node_embeddings, self.weights_pool)    # (B, S, K, D_in, D_out)
            bias = torch.einsum('b s d, d o -> b s o', node_embeddings, self.bias_pool)                  # (B, S, D_out)

            # Use the same sample-wise graph to propagate all time steps in the current sample.
            # The Chebyshev supports are computed on the fly and consumed immediately, without stacking them.
            x_gconv = torch.einsum('b l s i, b s i o -> b l s o', x, weights[:, :, 0])                   # T0, (B, L, S, D_out)
            if self.cheb_k >= 2:
                t0_x = x                                                                                   # (B, L, S, D_in)
                t1_x = torch.einsum('b s m, b l m i -> b l s i', g, x)                                    # (B, L, S, D_in)
                x_gconv = x_gconv + torch.einsum('b l s i, b s i o -> b l s o', t1_x, weights[:, :, 1])
                for k in range(2, self.cheb_k):
                    tk_x = 2 * torch.einsum('b s m, b l m i -> b l s i', g, t1_x) - t0_x                 # (B, L, S, D_in)
                    x_gconv = x_gconv + torch.einsum('b l s i, b s i o -> b l s o', tk_x, weights[:, :, k])
                    t0_x, t1_x = t1_x, tk_x
            x_gconv = x_gconv + bias.unsqueeze(1)
            return x_gconv

        # Static-only graph path for the w/o sample-wise adaptive graph ablation.
        # It uses the fixed support directly and does not build a sample-conditioned graph.
        if node_embeddings.dim() != 2:
            raise ValueError("Static graph path expects shared node embeddings with shape (S, D).")

        if static_prior is not None:
            g = static_prior / static_prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)                    # (S, S)
        else:
            g = torch.eye(node_embeddings.shape[0], device=x.device, dtype=x.dtype)                      # (S, S)

        weights = torch.einsum('s d, d k i o -> s k i o', node_embeddings, self.weights_pool)            # (S, K, D_in, D_out)
        bias = torch.einsum('s d, d o -> s o', node_embeddings, self.bias_pool)                          # (S, D_out)

        x_gconv = torch.einsum('b l s i, s i o -> b l s o', x, weights[:, 0])                            # T0, (B, L, S, D_out)
        if self.cheb_k >= 2:
            t0_x = x                                                                                      # (B, L, S, D_in)
            t1_x = torch.einsum('s m, b l m i -> b l s i', g, x)                                         # (B, L, S, D_in)
            x_gconv = x_gconv + torch.einsum('b l s i, s i o -> b l s o', t1_x, weights[:, 1])
            for k in range(2, self.cheb_k):
                tk_x = 2 * torch.einsum('s m, b l m i -> b l s i', g, t1_x) - t0_x                      # (B, L, S, D_in)
                x_gconv = x_gconv + torch.einsum('b l s i, s i o -> b l s o', tk_x, weights[:, k])
                t0_x, t1_x = t1_x, tk_x
        x_gconv = x_gconv + bias.unsqueeze(0).unsqueeze(0)
        return x_gconv


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: torch.Tensor, (B, L, S, D)
        :return: torch.Tensor, (B, L, S, D)
        """
        x_channel_first = x.permute(0, 3, 2, 1)                  # (B, D, S, L)
        hidden = self.fc2(self.drop(self.act(self.fc1(x_channel_first))))
        out = hidden + x_channel_first
        return out.permute(0, 3, 2, 1)                           # (B, L, S, D)


class AdaptiveSpatialBlock(nn.Module):
    def __init__(self, dim, spatial_embed_dim, cheb_k):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dgc = DynamicGraphConvolutionalLayer(dim, dim, spatial_embed_dim, cheb_k)
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, node_embeddings, static_prior=None, use_sample_wise_adaptive_graph=True):
        """
        :param x: torch.Tensor, (B, L, S, D)
        :return: torch.Tensor, (B, L, S, D)
        """
        spatial_out = self.dgc(
            self.norm(x),
            node_embeddings,
            static_prior=static_prior,
            use_sample_wise_adaptive_graph=use_sample_wise_adaptive_graph
        )                                                                                      # (B, L, S, D)
        gate_input = torch.cat([x, spatial_out], dim=-1)                                       # (B, L, S, 2D)
        gate = self.spatial_gate(gate_input)                                                   # (B, L, S, D)
        return x + gate * spatial_out


class SpatialTemporalMixerEncoder(nn.Module):
    def __init__(self, dim, spatial_embed_dim, cheb_k, depth, dropout=0.):
        super().__init__()
        self.pre_mlp = ResidualMLPBlock(dim=dim, dropout=dropout)
        self.spatial_blocks = nn.ModuleList(
            [AdaptiveSpatialBlock(dim=dim, spatial_embed_dim=spatial_embed_dim, cheb_k=cheb_k)
             for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [ResidualMLPBlock(dim=dim, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x, node_embeddings, static_prior=None, use_sample_wise_adaptive_graph=True):
        """
        :param x: torch.Tensor, (B, L, S, D)
        :return: x: torch.Tensor, (B, L, S, D)
        """
        x = self.pre_mlp(x)
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = spatial_block(
                x,
                node_embeddings,
                static_prior=static_prior,
                use_sample_wise_adaptive_graph=use_sample_wise_adaptive_graph
            )
            x = temporal_block(x)
        return x


class STBoost(nn.Module):
    def __init__(self, args, device, logger):
        super().__init__()
        assert args.p_len % 12 == 0 and args.q_len % 12 == 0, "p_len and q_len must be divisible by 12"
        self.args = args
        self.device = device
        self.logger = logger
        self.p_len = args.p_len
        self.q_len = args.q_len
        self.num_sensors = args.num_nodes
        self.duration = args.duration
        self.day_slots = (24 * 60) // self.duration
        self.period_mem_dim = args.stb_period_mem_dim
        self.dynamic_cheb_k = args.stb_dynamic_cheb_k
        self.static_cheb_k = args.stb_static_cheb_k
        self.dynamic_spatial_embed_dim = args.stb_dynamic_spatial_embed_dim
        self.num_hyper_nodes = args.stb_num_hyper_nodes
        self.static_spatial_embed_dim = args.stb_static_spatial_embed_dim
        self.time_embed_dim = args.stb_time_embed_dim

        self.st_mixer_depth = args.stb_st_mixer_depth
        if self.st_mixer_depth <= 0:
            raise ValueError("ST mixer depth must be greater than 0")
        self.temporal_embed_dim = args.stb_temporal_embed_dim
        self.temporal_dropout = args.stb_temporal_dropout

        self.cur_cl_len = 0
        self.use_curriculum_learning = args.stb_use_curriculum_learning

        self.use_tod = bool(args.stb_use_tod)
        self.use_dow = bool(args.stb_use_dow)
        self.use_input_temporal = bool(args.stb_use_input_temporal)
        self.use_input_period = bool(args.stb_use_input_period)
        self.use_input_residual = bool(args.stb_use_input_residual)
        self.use_period_memorizer = bool(args.stb_use_period_memorizer)
        self.sidecar_mode = args.stb_sidecar_mode
        self.mlp_hidden_dim = args.stb_mlp_hidden_dim
        self.use_sample_wise_adaptive_graph = bool(args.stb_use_sample_wise_adaptive_graph)
        self.use_static_graph = bool(args.stb_use_static_graph)
        self.use_spatial_modeling = bool(args.stb_use_spatial_modeling)
        self.use_base_prediction = bool(args.stb_use_base_prediction)
        self.use_static_hidden_concat = (
            self.sidecar_mode == 'full' and self.use_spatial_modeling and self.use_static_graph
        )

        if not (self.use_input_temporal or self.use_input_period or self.use_input_residual):
            raise ValueError("At least one of temporal/period/residual input channels must be enabled.")
        if not self.use_period_memorizer and (self.use_input_period or self.use_input_residual):
            raise ValueError("Disabling PPM also disables period and residual inputs. Use temporal input only.")
        if self.sidecar_mode == 'full' and self.use_spatial_modeling and not (
                self.use_sample_wise_adaptive_graph or self.use_static_graph):
            raise ValueError("Spatial modeling requires sample-wise adaptive graph or static graph.")
        if self.sidecar_mode == 'mlp' and self.mlp_hidden_dim <= 0:
            raise ValueError("MLP sidecar hidden dimension must be greater than 0.")

        # Convert the sampling interval into the number of discrete time steps per day.
        self.step_per_day = int(24 * 60 / self.duration)

        # Temporarily switch to the fixed STBoost seed before initializing its modules.
        self.logger.info("Temporarily switching to the fixed STBoost initialization seed...")

        # Save the current random state first, then restore it after STBoost initialization.
        rng_cpu_state = torch.get_rng_state()
        rng_gpu_state = None
        if torch.cuda.is_available():
            rng_gpu_state = torch.cuda.get_rng_state_all()

        torch.manual_seed(args.stb_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.stb_init_seed)

        # Define the semantic signal dimensions used by the temporal and graph pathways.
        self.temporal_signal_dim = \
            (1 if self.use_input_temporal else 0) + \
            (1 if self.use_input_period else 0) + \
            (1 if self.use_input_residual else 0)
        self.semantic_time_dim = \
            (2 if self.use_tod else 0) + \
            (2 if self.use_dow else 0)
        self.series_signal_dim = self.temporal_signal_dim + self.semantic_time_dim

        # Model
        self.period_memorizer = None

        # Project the configured temporal pathway (temporal + period by default) into the hidden space.
        self.temporal_path_len = self.p_len + self.q_len if self.use_base_prediction else self.p_len
        self.series_emb_layer = nn.Conv2d(
            in_channels=self.temporal_path_len * self.series_signal_dim,
            out_channels=self.temporal_embed_dim,
            kernel_size=1,
            bias=True
        )

        self.graph_summary_proj = None
        self.hyper_node_memory = None
        self.static_to_dynamic_proj = None
        self.gcn = None
        self.latest_debug_tensors = None
        self.node_prior_logit = None
        self.static_hidden_dim = 0
        self.hidden_dim = self.temporal_embed_dim
        self.st_mixer_encoder = None
        self.mlp_sidecar = None
        if self.sidecar_mode == 'full':
            # Build graph-conditioning node states from historical semantic summaries.
            self.graph_summary_proj = nn.Linear(self.series_signal_dim, self.temporal_embed_dim, bias=True)
            if self.use_spatial_modeling and self.use_sample_wise_adaptive_graph:
                self.hyper_node_memory = HyperNodeMemory(
                    input_dim=self.temporal_embed_dim,
                    embed_dim=self.dynamic_spatial_embed_dim,
                    num_hyper_nodes=self.num_hyper_nodes
                )
            if self.use_spatial_modeling and self.use_static_graph:
                self.static_to_dynamic_proj = nn.Linear(
                    self.static_spatial_embed_dim,
                    self.dynamic_spatial_embed_dim,
                    bias=True
                )
                self.gcn = GraphConvolutionalNetwork(
                    num_nodes=self.num_sensors,
                    embed_dim=self.static_spatial_embed_dim,
                    k=self.static_cheb_k,
                    hidden_dim=self.args.stb_static_hid_dim,
                    dropout=0.1
                )
            self.node_prior_logit = nn.Parameter(torch.tensor(-2.0))
            self.static_hidden_dim = self.static_spatial_embed_dim if self.use_static_hidden_concat else 0
            self.hidden_dim = self.temporal_embed_dim + self.static_hidden_dim
            if self.use_spatial_modeling:
                self.st_mixer_encoder = SpatialTemporalMixerEncoder(
                    dim=self.hidden_dim,
                    spatial_embed_dim=self.dynamic_spatial_embed_dim,
                    cheb_k=self.dynamic_cheb_k,
                    depth=self.st_mixer_depth,
                    dropout=self.temporal_dropout
                )
        else:
            self.mlp_sidecar = nn.Sequential(
                nn.Conv2d(self.temporal_embed_dim, self.mlp_hidden_dim, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Dropout(self.temporal_dropout),
                nn.Conv2d(self.mlp_hidden_dim, self.temporal_embed_dim, kernel_size=1, bias=True)
            )
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.q_len,
            kernel_size=1,
            bias=True
        )

        # Restore the original random state so the remaining pipeline keeps its original behavior.
        torch.set_rng_state(rng_cpu_state)
        if torch.cuda.is_available() and rng_gpu_state is not None:
            torch.cuda.set_rng_state_all(rng_gpu_state)

        self.logger.info(
            f"STBoost initialized with Seed {args.stb_init_seed}. Restored context to the base-model RNG state."
        )

    def process_sd_mx_for_gcn(self, sd_mx, epsilon=1e-3):
        """
        Convert distance matrix (with inf for unreachable nodes) into adjacency matrix for GCN.

        :param sd_mx: torch.Tensor, (num_nodes, num_nodes), distance matrix (may contain inf), on cuda
        :param epsilon: float, threshold for pruning small weights
        :return: torch.Tensor, adjacency matrix (num_nodes, num_nodes), on same device
        """
        device = sd_mx.device
        adj = sd_mx.clone()
        valid_mask = ~torch.isinf(adj)
        distances = adj[valid_mask]
        std = distances.std()
        if std.item() == 0:
            std = torch.tensor(1.0, device=device)
        adj = torch.exp(-torch.square(adj / std))
        adj[~valid_mask] = 0.0
        adj = torch.where(adj < epsilon, torch.zeros_like(adj), adj)
        return adj

    def forward(self, x, y_base, time_encoding, time_info, dist_adj, batches_seen, x_missing_mask=None):
        """

        :param x: (B, L, S, F), F denote the feature, for instance, traffic flow
        :param y_base: (B, base_q_len, S, 1), optional base-model future signal
        :param time_encoding: (B, P + Q, num_encoding(3))
        :param time_info: (B, P + Q, 2)
        :param dist_adj: (S, S)
        :param batches_seen: int
        :return: (B, Q, S, F)
        """
        time_info = torch.as_tensor(time_info, device=x.device)
        self.latest_debug_tensors = None

        semantic_time_parts = []
        if self.use_tod:
            timestep_index = time_encoding[..., 0].to(dtype=x.dtype)
            timestep_omega = 2 * math.pi * timestep_index / self.step_per_day
            timestep_cycle_encoding = torch.stack(
                [torch.sin(timestep_omega), torch.cos(timestep_omega)],
                dim=-1
            )                                                                                   # (B, P + Q, 2)
            semantic_time_parts.append(timestep_cycle_encoding)

        if self.use_dow:
            day_of_week_index = time_encoding[..., 1].to(dtype=x.dtype)
            day_of_week_omega = 2 * math.pi * day_of_week_index / 7.0
            day_of_week_cycle_encoding = torch.stack(
                [torch.sin(day_of_week_omega), torch.cos(day_of_week_omega)],
                dim=-1
            )                                                                                   # (B, P + Q, 2)
            semantic_time_parts.append(day_of_week_cycle_encoding)

        semantic_time_encoding = None
        if len(semantic_time_parts) > 0:
            semantic_time_encoding = torch.cat(semantic_time_parts, dim=-1)                     # (B, P + Q, D_semantic)

        gcn_adj = None
        static_node_embeddings = None
        static_dynamic_prior = None
        if self.sidecar_mode == 'full' and self.use_spatial_modeling and self.use_static_graph:
            gcn_adj = self.process_sd_mx_for_gcn(dist_adj)                                      # (S, S)
            static_node_embeddings = self.gcn(gcn_adj)                                          # (S, D_static)
            static_dynamic_prior = self.static_to_dynamic_proj(static_node_embeddings)          # (S, D_dynamic)

        if self.use_period_memorizer:
            assert self.period_memorizer is not None, "Please set Period Pattern Memorizer module first!"
            xy_period = self.period_memorizer.get_pattern_features(time_info)                   # (B, P + Q, S, 1)
        else:
            xy_period = x.new_zeros(x.shape[0], self.p_len + self.q_len, x.shape[2], 1)         # (B, P + Q, S, 1)
        x_period = xy_period[:, : self.p_len]                                                   # (B, P, S, 1)
        y_period = xy_period[:, self.p_len :]                                                   # (B, Q, S, 1)
        assert x_missing_mask is not None, "x_missing_mask must be provided from raw extra_inputs."
        x_temporal_fill = x_period if self.use_period_memorizer else torch.zeros_like(x)
        x_temporal = torch.where(x_missing_mask, x_temporal_fill, x)                            # (B, P, S, 1)
        x_residual = torch.where(
            x_missing_mask,
            torch.zeros_like(x),
            x - x_period
        )                                                                                       # (B, P, S, 1)

        y_temporal = x.new_zeros(x.shape[0], 0, x.shape[2], 1)
        y_residual = x.new_zeros(x.shape[0], 0, x.shape[2], 1)
        use_future_path = y_base is not None and y_base.shape[1] > 0
        if use_future_path:
            base_q_len = y_base.shape[1]
            if base_q_len >= self.q_len:
                y_temporal = y_base[:, : self.q_len]
            else:
                y_temporal_padding = y_period[:, base_q_len:]
                y_temporal = torch.cat([y_base, y_temporal_padding], dim=1)

            y_residual = torch.zeros_like(y_period).to(x.device)

            # Compose the P + Q pathway when the base model provides future temporal hints.
            series_temporal = torch.cat([x_temporal, y_temporal], dim=1)                        # (B, P + Q, S, 1)
            series_period = xy_period                                                           # (B, P + Q, S, 1)
            series_residual = torch.cat([x_residual, y_residual], dim=1)                        # (B, P + Q, S, 1)
            series_time_encoding = semantic_time_encoding                                       # (B, P + Q, D_semantic)
        else:
            # STBoost-only ablation: predict Q directly from the historical P window.
            series_temporal = x_temporal                                                        # (B, P, S, 1)
            series_period = x_period                                                            # (B, P, S, 1)
            series_residual = x_residual                                                        # (B, P, S, 1)
            series_time_encoding = (
                semantic_time_encoding[:, : self.p_len] if semantic_time_encoding is not None else None
            )                                                                                    # (B, P, D_semantic)

        signal_parts = []
        if self.use_input_temporal:
            signal_parts.append(series_temporal)
        if self.use_input_period:
            signal_parts.append(series_period)
        if self.use_input_residual:
            signal_parts.append(series_residual)
        if series_time_encoding is not None:
            time_embeddings = series_time_encoding[:, :, None, :].expand(-1, -1, x.shape[2], -1)     # (B, L, S, D_semantic)
            signal_parts.append(time_embeddings)
        signal_series = torch.cat(signal_parts, dim=-1)                                             # (B, L, S, D_signal)

        graph_node_embeddings = None
        hyper_query_input = None
        hyper_attention = None
        static_graph_embeddings = None
        node_hidden_summary = None
        if self.sidecar_mode == 'full' and self.use_spatial_modeling:
            node_prior_ratio = torch.sigmoid(self.node_prior_logit)
            if self.use_sample_wise_adaptive_graph:
                history_signal_series = signal_series[:, : self.p_len]                              # (B, P, S, D_signal)
                history_signal_emb = self.graph_summary_proj(history_signal_series)                 # (B, P, S, D_temporal)
                node_hidden_summary = history_signal_emb.mean(dim=1)                                # (B, S, D_temporal)

                hyper_query_input = node_hidden_summary
                hyper_node_embeddings, hyper_attention = self.hyper_node_memory(hyper_query_input)  # (B, S, D_dynamic), (B, S, M)
                graph_node_embeddings = hyper_node_embeddings
                if self.use_static_graph:
                    static_graph_embeddings = torch.tanh(static_dynamic_prior).unsqueeze(0).expand(x.shape[0], -1, -1)
                    graph_node_embeddings = graph_node_embeddings + node_prior_ratio * static_graph_embeddings
            else:
                assert self.use_static_graph, "Static graph must be enabled when sample-wise adaptive graph is disabled."
                graph_node_embeddings = torch.tanh(static_dynamic_prior)                            # (S, D_dynamic)
                static_graph_embeddings = graph_node_embeddings.unsqueeze(0).expand(x.shape[0], -1, -1)
        # Project the selected temporal pathway into the main representation used by the predictor.
        temporal_path_len = signal_series.shape[1]
        if temporal_path_len != self.temporal_path_len:
            raise ValueError(
                f"Unsupported temporal pathway length: {temporal_path_len}. "
                f"Expected {self.temporal_path_len} for the current STBoost configuration."
            )
        signal_series = rearrange(signal_series, 'b l s c -> b (l c) s 1')                          # (B, L*D_signal, S, 1)
        time_series_emb = self.series_emb_layer(signal_series)                                       # (B, D_temporal, S, 1)

        static_hidden_emb = None
        if self.sidecar_mode == 'mlp':
            hidden = self.mlp_sidecar(time_series_emb)                                               # (B, D_temporal, S, 1)
            hidden = hidden.permute(0, 3, 2, 1)                                                      # (B, 1, S, D_temporal)
        else:
            if self.use_static_hidden_concat:
                static_hidden_emb = static_node_embeddings.transpose(0, 1).unsqueeze(0).unsqueeze(-1)    # (1, D_static, S, 1)
                static_hidden_emb = static_hidden_emb.expand(x.shape[0], -1, -1, -1)                     # (B, D_static, S, 1)
                hidden = torch.cat([time_series_emb, static_hidden_emb], dim=1)                          # (B, D_hidden, S, 1)
            else:
                # Disable all static hidden states for static-graph or spatial-modeling ablations.
                hidden = time_series_emb                                                                 # (B, D_temporal, S, 1)
            hidden = hidden.permute(0, 3, 2, 1)                                                           # (B, 1, S, D_hidden)

            if self.use_spatial_modeling:
                hidden = self.st_mixer_encoder(
                    hidden,
                    graph_node_embeddings,
                    static_prior=gcn_adj,
                    use_sample_wise_adaptive_graph=self.use_sample_wise_adaptive_graph
                )                                                                                         # (B, 1, S, D_hidden)

        if self.args.stb_dump_debug_arrays:
            self.latest_debug_tensors = {
                'x_temporal': x_temporal.detach(),
                'x_period': x_period.detach(),
                'x_residual': x_residual.detach(),
                'y_temporal': y_temporal.detach(),
                'y_period': y_period.detach(),
                'y_residual': y_residual.detach(),
                'x_missing_mask': x_missing_mask.detach(),
                'time_series_emb': time_series_emb.detach(),
                'hidden': hidden.detach()
            }
            if static_hidden_emb is not None:
                self.latest_debug_tensors['static_hidden_emb'] = static_hidden_emb.detach()
            if node_hidden_summary is not None:
                self.latest_debug_tensors['node_hidden_summary'] = node_hidden_summary.detach()
            if static_graph_embeddings is not None:
                self.latest_debug_tensors['static_graph_embeddings'] = static_graph_embeddings.detach()
            if graph_node_embeddings is not None:
                self.latest_debug_tensors['graph_node_embeddings'] = graph_node_embeddings.detach()
            if hyper_query_input is not None:
                self.latest_debug_tensors['hyper_query_input'] = hyper_query_input.detach()
            if hyper_attention is not None:
                self.latest_debug_tensors['hyper_attention'] = hyper_attention.detach()

        y_pred = self.regression_layer(hidden.permute(0, 3, 2, 1))                                       # (B, Q, S, 1)

        if self.training and self.use_curriculum_learning:
            if batches_seen % self.args.stb_step_size == 0 and self.cur_cl_len < self.q_len:
                self.cur_cl_len += 1
                self.logger.info(f'Current task length increase from {self.cur_cl_len - 1} to {self.cur_cl_len}')
                self.logger.info(f'Current batches_seen is {batches_seen}')
            return y_pred[:, : self.cur_cl_len]
        else:
            return y_pred


class ParallelStructure(nn.Module):
    def __init__(self, base_model, booster, args, logger, base_p_len, base_q_len, base_model_name):
        super().__init__()
        self.base_model = base_model
        self.booster = booster
        self.base_model_name = base_model_name
        self.args = args
        self.logger = logger
        self.base_p_len = base_p_len
        self.base_q_len = base_q_len
        self.use_base_prediction = bool(args.stb_use_base_prediction)

        self.stage = None
        self.unified_training = False
        unified_init_logit = float(getattr(args, 'stb_unified_init_logit', 4.0))
        self.weight = nn.Parameter(unified_init_logit * torch.ones(1, base_q_len, args.num_nodes, 1))
        self.latest_stage2_debug = None

    def set_unified_training(self):
        if not self.unified_training:
            self.unified_training = True

    def get_unified_training(self):
        return self.unified_training

    def set_period_pattern_memorizer(self, module):
        if not self.booster.use_period_memorizer:
            raise RuntimeError("Period Pattern Memorizer is disabled for this STBoost run!")
        if self.booster.period_memorizer is None:
            self.booster.period_memorizer = module
        else:
            raise RuntimeError("Period Pattern Memorizer module has already set!")

    def should_fuse_with_base(self):
        return self.use_base_prediction and self.unified_training

    def _finalize_stage2_output(self, y_base, y_boost):
        if self.should_fuse_with_base():
            delta = y_boost[:, : self.base_q_len] - y_base
            weight = torch.sigmoid(self.weight)
            y_out_short = y_base + weight * delta
            y_out_long = y_boost[:, self.base_q_len:]
            y_out = torch.cat([y_out_short, y_out_long], dim=1)
        else:   # only boost
            y_out = y_boost

        if self.booster.latest_debug_tensors is not None:
            if y_base is None:
                y_base_debug = y_boost.new_zeros(
                    y_boost.shape[0], self.base_q_len, y_boost.shape[2], y_boost.shape[3]
                )
            else:
                y_base_debug = y_base
            self.latest_stage2_debug = {
                'y_base': y_base_debug.detach(),
                'y_boost': y_boost.detach(),
                'y_out': y_out.detach(),
                'booster_debug': self.booster.latest_debug_tensors
            }
        else:
            self.latest_stage2_debug = None
        return y_out

    def set_stage(self, stage):
        self.stage = stage
        if stage == 1:
            for param in self.booster.parameters():
                param.requires_grad = False
            self.weight.requires_grad = False
            for param in self.base_model.parameters():
                param.requires_grad = True
        elif stage == 2:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.booster.parameters():
                param.requires_grad = True
            if self.booster.period_memorizer is not None:
                self.booster.period_memorizer.temp_pattern_memory.requires_grad = False  # this module do not need to train
            self.weight.requires_grad = self.use_base_prediction
        else:
            raise ValueError("Stage must be 1 or 2")

    def forward(self, x, *args, **kwargs):
        """
        :param x: (B, P, S, F), F denote the feature, for instance, traffic flow
        :return output: (B, Q, S, F)
        """
        assert self.stage is not None, "stage must be initialized"
        if self.stage == 1:
            # short-term modeling using base model
            if self.base_model_name == "MegaCRN":
                """

                :param base model input:  (B, P, S, D_in)
                :param base model output: (B, Q, S, D_out)
                :return output: (B, Q, S, 1)
                """
                ycov = kwargs.get('ycov', args[0] if len(args) >= 1 else None)
                y = kwargs.get('y', args[1] if len(args) >= 2 else None)
                batches_seen = kwargs.get('batches_seen', args[2] if len(args) >= 3 else None)

                if y is not None and batches_seen is not None:
                    output, h_att, query, pos, neg = self.base_model(x, ycov, y, batches_seen)
                else:
                    output, h_att, query, pos, neg = self.base_model(x, ycov)

                return output, h_att, query, pos, neg
            else:
                raise NotImplementedError("Unsupported model!")
        elif self.stage == 2:
            if self.base_model_name == "MegaCRN":
                """
                :param x: (B, P, S, D_in)   D_in = 1
                :param time_encoding: (B, P + Q, 3), torch.Tensor, raw calendar indices of x and y
                :param time_info: (B, P + Q, 2)
                :param dist_adj: (S, S)
                :return y_out: (B, Q, S, 1)
                """
                if self.booster.use_period_memorizer:
                    assert self.booster.period_memorizer is not None, "Please set Period Pattern Memorizer module first!"
                ycov = kwargs.get('ycov', args[0] if len(args) >= 1 else None)
                time_encoding = kwargs.get('time_encoding', args[1] if len(args) >= 2 else None)
                time_info = kwargs.get('time_info', args[2] if len(args) >= 3 else None)
                dist_adj = kwargs.get('dist_adj', args[3] if len(args) >= 4 else None)
                stb_batches_seen = kwargs.get('stb_batches_seen', args[4] if len(args) >= 5 else None)
                x_missing_mask = kwargs.get('x_missing_mask', args[5] if len(args) >= 6 else None)
                y_base_cache = kwargs.get('y_base_cache', None)

                if self.use_base_prediction:
                    if y_base_cache is not None:
                        y_base = y_base_cache
                    else:
                        # Base model input process
                        x_base = x[:, -self.base_p_len:]
                        if getattr(self.args, 'stb_stage2_align_inputs', 0):
                            x_base = convert_standardized_scale(x_base, self.stage2_scaler, self.base_scaler)
                        ycov_base = ycov[:, : self.base_q_len]
                        y_base, *_ = self.base_model(x_base, ycov_base)  # (B, base_horizon, S, 1)
                        if getattr(self.args, 'stb_stage2_align_inputs', 0):
                            y_base = convert_standardized_scale(y_base, self.base_scaler, self.stage2_scaler)
                else:
                    y_base = None

                # Boost predict
                y_boost = self.booster(
                    x, y_base, time_encoding, time_info, dist_adj, stb_batches_seen,
                    x_missing_mask=x_missing_mask
                )  # (B, Q, S, 1)

                return self._finalize_stage2_output(y_base, y_boost)
            else:
                raise NotImplementedError("Unsupported model!")
        else:
            raise NotImplementedError("Unknown stage: {}".format(self.stage))
