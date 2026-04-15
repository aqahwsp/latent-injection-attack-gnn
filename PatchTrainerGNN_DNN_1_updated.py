import copy
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import (
    coalesce,
    dense_to_sparse,
    k_hop_subgraph,
    remove_self_loops,
    to_dense_batch,
    to_undirected,
)


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_p: Tensor) -> Tensor:
        ctx.save_for_backward(input_p)
        return (input_p >= 0.5).float()

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        (p,) = ctx.saved_tensors
        w = torch.clamp(1 + 5 * (1 - (p - 0.5).abs() / 0.1), min=1.0)
        return grad * w


def ste_sigmoid_binary(x: Tensor) -> Tensor:
    return StraightThroughEstimator.apply(torch.sigmoid(x))


def _forward_clean_model(model: nn.Module, x: Tensor, edge_index: Tensor) -> Tensor:
    try:
        return model(x, edge_index)
    except TypeError:
        return model(x)


def _to_undirected_edge_index_only(edge_index: Tensor, num_nodes: int) -> Tensor:
    result = to_undirected(edge_index, num_nodes=int(num_nodes))
    if isinstance(result, tuple):
        edge_index = result[0]
    else:
        edge_index = result
    return edge_index.long()


def _coalesce_edge_index_only(edge_index: Tensor, num_nodes: int) -> Tensor:
    result = coalesce(edge_index, None, int(num_nodes), int(num_nodes))
    if isinstance(result, tuple):
        edge_index = result[0]
    else:
        edge_index = result
    return edge_index.long()


def _finalize_binary_undirected_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0), dtype=torch.long)

    edge_index = edge_index.long()
    edge_index = _to_undirected_edge_index_only(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = _coalesce_edge_index_only(edge_index, num_nodes)
    return edge_index.long()
def _parse_amp_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = "float16" if dtype is None else str(dtype).lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"不支持的 amp_dtype: {dtype}")


def _enable_fast_attention_backends() -> None:
    if (not torch.cuda.is_available()) or (not hasattr(torch.backends, "cuda")):
        return

    for fn_name in ["enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"]:
        fn = getattr(torch.backends.cuda, fn_name, None)
        if callable(fn):
            try:
                fn(True)
            except Exception:
                pass

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    try:
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPatchGenerator(nn.Module):
    """
    支持 batch 的共享生成器：
    - 输入：多个目标节点对应的领域子图，打包成一个 PyG Batch
    - 输出：一次性输出多个目标节点的 patch
    """

    def __init__(
        self,
        num_node_features: int,
        d_model: int = 192,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_hidden = max(d_model // 2, 48)

        self.encoder_conv1 = GATConv(
            num_node_features,
            encoder_hidden,
            heads=4,
            dropout=dropout,
        )
        self.encoder_activation = nn.ELU()
        self.encoder_conv2 = GATConv(
            encoder_hidden * 4,
            d_model,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.feature_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_node_features),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        subgraph_x: Tensor,
        subgraph_edge_index: Tensor,
        target_local_idx,
        num_patch_nodes: int,
        batch: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x = self.encoder_conv1(subgraph_x, subgraph_edge_index)
        x = self.encoder_activation(x)
        memory_nodes = self.encoder_conv2(x, subgraph_edge_index)

        single_graph_input = (
            batch is None
            and ptr is None
            and (
                isinstance(target_local_idx, int)
                or (torch.is_tensor(target_local_idx) and int(target_local_idx.numel()) == 1)
            )
        )

        if batch is None:
            batch = torch.zeros(
                memory_nodes.size(0),
                dtype=torch.long,
                device=memory_nodes.device,
            )

        if not torch.is_tensor(target_local_idx):
            target_local_idx = torch.as_tensor(
                target_local_idx,
                device=memory_nodes.device,
                dtype=torch.long,
            )
        target_local_idx = target_local_idx.view(-1).long()

        memory_dense, memory_valid_mask = to_dense_batch(memory_nodes, batch=batch)
        batch_size = int(memory_dense.size(0))

        if int(target_local_idx.numel()) != batch_size:
            raise ValueError(
                f"target_local_idx 数量与 batch 大小不一致："
                f"{int(target_local_idx.numel())} vs {batch_size}"
            )

        batch_arange = torch.arange(batch_size, device=memory_nodes.device)
        tgt = memory_dense[batch_arange, target_local_idx].unsqueeze(1)

        generated_features_logits_list = []
        generated_connections_logits_list = []

        memory_key_padding_mask = ~memory_valid_mask

        for _ in range(int(num_patch_nodes)):
            tgt_with_pos = self.pos_encoder(tgt)
            seq_len = int(tgt_with_pos.size(1))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

            if (
                    memory_nodes.is_cuda
                    and hasattr(torch.backends, "cuda")
                    and hasattr(torch.backends.cuda, "sdp_kernel")
            ):
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_mem_efficient=True,
                        enable_math=True,
                ):
                    output = self.transformer_decoder(
                        tgt=tgt_with_pos,
                        memory=memory_dense,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )
            else:
                output = self.transformer_decoder(
                    tgt=tgt_with_pos,
                    memory=memory_dense,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            last_output = output[:, -1, :]

            new_feature_logits = self.feature_head(last_output)
            generated_features_logits_list.append(new_feature_logits)

            connection_logits = torch.bmm(output, last_output.unsqueeze(-1)).squeeze(-1)
            generated_connections_logits_list.append(connection_logits)

            tgt = torch.cat([tgt, last_output.unsqueeze(1).detach()], dim=1)

        final_adj_logits = torch.zeros(
            (batch_size, int(num_patch_nodes) + 1, int(num_patch_nodes) + 1),
            device=subgraph_x.device,
        )

        for i in range(int(num_patch_nodes)):
            connection_logits = generated_connections_logits_list[i]
            target_connection_logit = connection_logits[:, 0]
            final_adj_logits[:, i, int(num_patch_nodes)] = target_connection_logit
            final_adj_logits[:, int(num_patch_nodes), i] = target_connection_logit

            if i > 0:
                patch_connection_logits = connection_logits[:, 1 : i + 1]
                final_adj_logits[:, i, :i] = patch_connection_logits
                final_adj_logits[:, :i, i] = patch_connection_logits

        final_features_logits = torch.stack(generated_features_logits_list, dim=1)
        final_adj_logits = torch.triu(final_adj_logits, diagonal=1)
        final_adj_logits = final_adj_logits + final_adj_logits.transpose(-1, -2)
        final_adj_logits.diagonal(dim1=-2, dim2=-1).zero_()

        if single_graph_input and batch_size == 1:
            return final_features_logits.squeeze(0), final_adj_logits.squeeze(0)
        return final_features_logits, final_adj_logits


class PatchTrainerGNN:
    def __init__(
        self,
        clean_gat_model: Optional[nn.Module] = None,
        num_patch_nodes: int = 40,
        num_trigger_nodes: int = 20,
        num_node_features: int = 1433,
        num_classes: int = 7,
        device: str = "cpu",
        lr: float = 2e-5,
        d_model: int = 192,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        weight_decay: float = 5e-6,
        defense_node_drop_prob: float = 0.9,
        defense_edge_drop_prob: float = 0.8,
        binary_logit_scale: float = 4.0,
        target_region_hops: int = 1,
        defense_trials: int = 24,
        defense_drop_prob: float = 0.99,
        generator_subgraph_hops: int = 1,
        shared_generator: bool = True,
        clean_model: Optional[nn.Module] = None,
        default_train_batch_size: int = 4,
        amp_enabled: bool = True,
        amp_dtype: str = "float16",
        freeze_patch_during_trigger_finetune: bool = True,
        _region_nodes_cache=None,
        **kwargs,
    ):
        if clean_model is not None and clean_gat_model is None:
            clean_gat_model = clean_model
        if clean_gat_model is None:
            raise ValueError("必须提供 clean_gat_model 或 clean_model。")

        self.device = torch.device(device)
        self.clean_gat_model = clean_gat_model.to(self.device).eval()
        self.clean_model = self.clean_gat_model

        for p in self.clean_gat_model.parameters():
            p.requires_grad_(False)

        self.num_patch_nodes = int(num_patch_nodes)
        self.num_trigger_nodes = int(num_trigger_nodes)
        self.num_node_features = int(num_node_features)
        self.num_classes = int(num_classes)

        self.target_region_hops = int(target_region_hops)
        self.defense_trials = int(defense_trials)
        self.defense_drop_prob = float(defense_drop_prob)
        self.defense_node_drop_prob = float(defense_node_drop_prob)
        self.defense_edge_drop_prob = float(defense_edge_drop_prob)
        self.binary_logit_scale = float(binary_logit_scale)

        self.generator_subgraph_hops = int(generator_subgraph_hops)
        self.shared_generator = bool(shared_generator)
        self.default_train_batch_size = int(default_train_batch_size)
        self.amp_enabled = bool(amp_enabled and self.device.type == "cuda")
        self.amp_dtype = _parse_amp_dtype(amp_dtype)
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.freeze_patch_during_trigger_finetune = bool(freeze_patch_during_trigger_finetune)
        if self.device.type == "cuda":
            _enable_fast_attention_backends()

        self.patch_generator = TransformerPatchGenerator(
            num_node_features=self.num_node_features,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.patch_generator.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        self.best_patch_adj_logits = None
        self.best_patch_node_features_logits = None
        self.best_patch_adj = None
        self.best_patch_node_features = None

        self.best_trigger_feature_logits = None
        self.best_trigger_adj_logits_rows = None

        self.best_full_patch_adj_logits = None
        self.best_full_patch_node_features_logits = None
        self.best_full_patch_adj = None
        self.best_full_patch_node_features = None
        self.stage1_frozen_patch_generator = None
        self._region_nodes_cache = {} if _region_nodes_cache is None else _region_nodes_cache

    @staticmethod
    def _cpu_clone(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().clone()

    def release_unused_cache(self, keep_best=False):
        cache_names = []

        if not keep_best:
            cache_names.extend(
                [
                    "best_patch_adj_logits",
                    "best_patch_node_features_logits",
                    "best_patch_adj",
                    "best_patch_node_features",
                    "best_trigger_feature_logits",
                    "best_trigger_adj_logits_rows",
                    "best_full_patch_adj_logits",
                    "best_full_patch_node_features_logits",
                    "best_full_patch_adj",
                    "best_full_patch_node_features",
                    "stage1_frozen_patch_generator",
                ]
            )

        for name in cache_names:
            if hasattr(self, name):
                setattr(self, name, None)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _backward_step_patch_generator(self, loss: Tensor, clip_grad_norm: float) -> None:
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.patch_generator.parameters(),
                float(clip_grad_norm),
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.patch_generator.parameters(),
                float(clip_grad_norm),
            )
            self.optimizer.step()

    @staticmethod
    def _sanitize_binary_adj(adj_binary: Tensor) -> Tensor:
        adj_binary = (adj_binary >= 0.5).float()
        triu = torch.triu(adj_binary, diagonal=1)
        adj_binary = triu + triu.transpose(-1, -2)
        adj_binary.diagonal(dim1=-2, dim2=-1).zero_()
        return adj_binary

    def _binary_to_logits(self, binary_tensor: Tensor) -> Tensor:
        positive = torch.full_like(binary_tensor, self.binary_logit_scale)
        negative = torch.full_like(binary_tensor, -self.binary_logit_scale)
        return torch.where(binary_tensor >= 0.5, positive, negative)

    def _hard_binarize_patch_outputs(
        self,
        patch_adj_logits: Tensor,
        patch_features_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        patch_adj_binary = self._sanitize_binary_adj(torch.sigmoid(patch_adj_logits))
        patch_features_binary = (torch.sigmoid(patch_features_logits) >= 0.5).float()
        return patch_adj_binary, patch_features_binary

    @staticmethod
    def _is_binary_tensor(tensor: Tensor, atol: float = 1e-6) -> bool:
        if tensor is None or (not torch.is_tensor(tensor)):
            return False
        t = tensor.detach()
        if not t.is_floating_point():
            t = t.float()
        in_range = torch.all((t >= -atol) & (t <= 1.0 + atol)).item()
        integer_like = torch.all((t - t.round()).abs() <= atol).item()
        return bool(in_range and integer_like)

    def _to_hard_feature_tensor(self, tensor: Tensor) -> Tensor:
        if self._is_binary_tensor(tensor):
            return (tensor >= 0.5).float()
        return ste_sigmoid_binary(tensor)

    def _ste_binary_adj(self, adj_logits: Tensor) -> Tensor:
        adj_binary = ste_sigmoid_binary(adj_logits)
        triu = torch.triu(adj_binary, diagonal=1)
        adj_binary = triu + triu.transpose(-1, -2)
        adj_binary.diagonal(dim1=-2, dim2=-1).zero_()
        return adj_binary

    def _to_hard_adj_tensor(self, tensor: Tensor) -> Tensor:
        if self._is_binary_tensor(tensor):
            return self._sanitize_binary_adj(tensor.float())
        return self._ste_binary_adj(tensor)

    def export_fixed_attack_patch_binary(
        self,
        patch_adj_logits: Tensor,
        patch_feature_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            return self._hard_binarize_patch_outputs(patch_adj_logits, patch_feature_logits)

    def _prepare_unique_undirected_edges(self, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return edge_index.new_empty((2, 0))
        row, col = edge_index
        mask = row < col
        return torch.stack([row[mask], col[mask]], dim=0)

    def _ensure_patch_feature_width(self, patch_node_features: Tensor) -> Tensor:
        if patch_node_features.dim() != 2:
            raise ValueError(
                f"patch_node_features 期望为二维张量 [num_patch_nodes, num_features]，实际为 {tuple(patch_node_features.shape)}。"
            )

        current_dim = int(patch_node_features.size(1))
        target_dim = int(self.num_node_features)
        if current_dim == target_dim:
            return patch_node_features

        if current_dim > target_dim:
            return patch_node_features[:, :target_dim]

        pad_width = target_dim - current_dim
        padding = torch.zeros(
            patch_node_features.size(0),
            pad_width,
            dtype=patch_node_features.dtype,
            device=patch_node_features.device,
        )
        return torch.cat([patch_node_features, padding], dim=1)

    def _validate_patch_tensor_shapes(
        self,
        patch_adj_tensor_for_structure: Tensor,
        patch_node_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        expected_patch_nodes = int(self.num_patch_nodes)
        expected_adj_size = expected_patch_nodes + 1

        if patch_adj_tensor_for_structure.dim() != 2:
            raise ValueError(
                f"patch_adj_tensor_for_structure 期望为二维张量，实际为 {tuple(patch_adj_tensor_for_structure.shape)}。"
            )
        if (
            patch_adj_tensor_for_structure.size(0) < expected_adj_size
            or patch_adj_tensor_for_structure.size(1) < expected_adj_size
        ):
            raise ValueError(
                f"patch_adj_tensor_for_structure 形状不足，至少需要 {(expected_adj_size, expected_adj_size)}，"
                f"实际为 {tuple(patch_adj_tensor_for_structure.shape)}。"
            )

        patch_node_features = self._ensure_patch_feature_width(patch_node_features)
        if patch_node_features.size(0) < expected_patch_nodes:
            raise ValueError(
                f"patch_node_features 节点数不足，至少需要 {expected_patch_nodes} 行，"
                f"实际为 {int(patch_node_features.size(0))} 行。"
            )

        patch_adj_tensor_for_structure = patch_adj_tensor_for_structure[:expected_adj_size, :expected_adj_size]
        patch_node_features = patch_node_features[:expected_patch_nodes]
        return patch_adj_tensor_for_structure, patch_node_features

    def build_target_subgraph(
        self,
        data,
        target_node_idx: int,
        num_hops: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int, Tensor]:
        num_hops = self.generator_subgraph_hops if num_hops is None else int(num_hops)
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=int(target_node_idx),
            num_hops=int(num_hops),
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=int(data.x.size(0)),
        )
        sub_x = data.x[subset]
        sub_edge_index = _finalize_binary_undirected_edge_index(
            sub_edge_index,
            int(sub_x.size(0)),
        ).to(sub_x.device)
        local_target_idx = int(mapping.view(-1)[0].item())
        return sub_x, sub_edge_index, local_target_idx, subset

    def build_target_subgraph_batch(
        self,
        data,
        target_node_indices: Sequence[int],
        num_hops: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        subgraph_data_list = []
        target_local_idx_list = []
        subset_list = []

        for node_idx in target_node_indices:
            sub_x, sub_edge_index, local_target_idx, subset = self.build_target_subgraph(
                data=data,
                target_node_idx=int(node_idx),
                num_hops=num_hops,
            )
            subgraph_data_list.append(Data(x=sub_x, edge_index=sub_edge_index))
            target_local_idx_list.append(int(local_target_idx))
            subset_list.append(subset)

        batch_obj = Batch.from_data_list(subgraph_data_list).to(self.device)
        target_local_idx_tensor = torch.tensor(
            target_local_idx_list,
            dtype=torch.long,
            device=self.device,
        )
        return (
            batch_obj.x,
            batch_obj.edge_index,
            batch_obj.batch,
            batch_obj.ptr,
            target_local_idx_tensor,
            subset_list,
        )

    def _generate_patch_batch_for_targets_with_generator(
        self,
        generator: nn.Module,
        data,
        target_node_indices: Sequence[int],
    ) -> Tuple[Tensor, Tensor]:
        (
            subgraph_x,
            subgraph_edge_index,
            subgraph_batch,
            subgraph_ptr,
            target_local_idx_tensor,
            _,
        ) = self.build_target_subgraph_batch(
            data=data,
            target_node_indices=target_node_indices,
            num_hops=self.generator_subgraph_hops,
        )

        patch_node_features_logits, patch_adj_logits = generator(
            subgraph_x=subgraph_x,
            subgraph_edge_index=subgraph_edge_index,
            target_local_idx=target_local_idx_tensor,
            num_patch_nodes=self.num_patch_nodes,
            batch=subgraph_batch,
            ptr=subgraph_ptr,
        )
        return patch_node_features_logits, patch_adj_logits

    def generate_patch_batch_for_targets(
        self,
        data,
        target_node_indices: Sequence[int],
    ) -> Tuple[Tensor, Tensor]:
        return self._generate_patch_batch_for_targets_with_generator(
            generator=self.patch_generator,
            data=data,
            target_node_indices=target_node_indices,
        )

    def generate_patch_for_target(
        self,
        data,
        target_node_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        patch_node_features_logits, patch_adj_logits = self.generate_patch_batch_for_targets(
            data=data,
            target_node_indices=[int(target_node_idx)],
        )
        return patch_node_features_logits.squeeze(0), patch_adj_logits.squeeze(0)

    def _refresh_stage1_frozen_patch_generator(self) -> None:
        self.stage1_frozen_patch_generator = copy.deepcopy(self.patch_generator).to(self.device)
        self.stage1_frozen_patch_generator.eval()
        for p in self.stage1_frozen_patch_generator.parameters():
            p.requires_grad_(False)

    def _ensure_stage1_frozen_patch_generator(self) -> None:
        if self.stage1_frozen_patch_generator is None:
            print("[WARN] 未找到第一阶段 patch 快照，自动使用当前 generator 创建冻结快照。")
            self._refresh_stage1_frozen_patch_generator()

    def generate_frozen_patch_batch_for_targets(
        self,
        data,
        target_node_indices: Sequence[int],
    ) -> Tuple[Tensor, Tensor]:
        self._ensure_stage1_frozen_patch_generator()
        with torch.no_grad():
            return self._generate_patch_batch_for_targets_with_generator(
                generator=self.stage1_frozen_patch_generator,
                data=data,
                target_node_indices=target_node_indices,
            )

    def generate_frozen_patch_for_target(
        self,
        data,
        target_node_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        patch_node_features_logits, patch_adj_logits = self.generate_frozen_patch_batch_for_targets(
            data=data,
            target_node_indices=[int(target_node_idx)],
        )
        return patch_node_features_logits.squeeze(0), patch_adj_logits.squeeze(0)

    def _merge_frozen_patch_with_current_trigger(
        self,
        current_patch_feature_logits_batch: Tensor,
        current_patch_adj_logits_batch: Tensor,
        frozen_patch_feature_logits_batch: Tensor,
        frozen_patch_adj_logits_batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        single_input = current_patch_adj_logits_batch.dim() == 2
        if single_input:
            current_patch_feature_logits_batch = current_patch_feature_logits_batch.unsqueeze(0)
            current_patch_adj_logits_batch = current_patch_adj_logits_batch.unsqueeze(0)
            frozen_patch_feature_logits_batch = frozen_patch_feature_logits_batch.unsqueeze(0)
            frozen_patch_adj_logits_batch = frozen_patch_adj_logits_batch.unsqueeze(0)

        frozen_patch_feature_logits_batch = frozen_patch_feature_logits_batch.detach().to(self.device)
        frozen_patch_adj_logits_batch = frozen_patch_adj_logits_batch.detach().to(self.device)

        if current_patch_feature_logits_batch.shape != frozen_patch_feature_logits_batch.shape:
            raise ValueError(
                f"current/frozen feature logits shape 不一致: "
                f"{tuple(current_patch_feature_logits_batch.shape)} vs "
                f"{tuple(frozen_patch_feature_logits_batch.shape)}"
            )
        if current_patch_adj_logits_batch.shape != frozen_patch_adj_logits_batch.shape:
            raise ValueError(
                f"current/frozen adj logits shape 不一致: "
                f"{tuple(current_patch_adj_logits_batch.shape)} vs "
                f"{tuple(frozen_patch_adj_logits_batch.shape)}"
            )

        feature_mask = torch.zeros_like(current_patch_feature_logits_batch)
        if self.num_trigger_nodes > 0:
            feature_mask[:, : self.num_trigger_nodes, :] = 1.0

        adj_mask = torch.zeros_like(current_patch_adj_logits_batch)
        if self.num_trigger_nodes > 0:
            # trigger-trigger
            adj_mask[:, : self.num_trigger_nodes, : self.num_trigger_nodes] = 1.0
            # trigger-patch
            adj_mask[:, : self.num_trigger_nodes, self.num_trigger_nodes : self.num_patch_nodes] = 1.0
            adj_mask[:, self.num_trigger_nodes : self.num_patch_nodes, : self.num_trigger_nodes] = 1.0
            # trigger-target
            adj_mask[:, : self.num_trigger_nodes, self.num_patch_nodes] = 1.0
            adj_mask[:, self.num_patch_nodes, : self.num_trigger_nodes] = 1.0

        merged_feature_logits = (
            feature_mask * current_patch_feature_logits_batch
            + (1.0 - feature_mask) * frozen_patch_feature_logits_batch
        )
        merged_adj_logits = (
            adj_mask * current_patch_adj_logits_batch
            + (1.0 - adj_mask) * frozen_patch_adj_logits_batch
        )

        merged_adj_logits = 0.5 * (
            merged_adj_logits + merged_adj_logits.transpose(-1, -2)
        )
        merged_adj_logits.diagonal(dim1=-2, dim2=-1).fill_(-self.binary_logit_scale)

        if single_input:
            return merged_feature_logits.squeeze(0), merged_adj_logits.squeeze(0)
        return merged_feature_logits, merged_adj_logits

    def _compose_full_patch_logits_for_targets(
        self,
        data,
        target_node_indices: Sequence[int],
    ) -> Tuple[Tensor, Tensor]:
        current_patch_feature_logits_batch, current_patch_adj_logits_batch = (
            self.generate_patch_batch_for_targets(
                data=data,
                target_node_indices=target_node_indices,
            )
        )

        if self.num_trigger_nodes <= 0 or (not self.freeze_patch_during_trigger_finetune):
            return current_patch_feature_logits_batch, current_patch_adj_logits_batch

        self._ensure_stage1_frozen_patch_generator()

        with torch.no_grad():
            frozen_patch_feature_logits_batch, frozen_patch_adj_logits_batch = (
                self._generate_patch_batch_for_targets_with_generator(
                    generator=self.stage1_frozen_patch_generator,
                    data=data,
                    target_node_indices=target_node_indices,
                )
            )

        return self._merge_frozen_patch_with_current_trigger(
            current_patch_feature_logits_batch=current_patch_feature_logits_batch,
            current_patch_adj_logits_batch=current_patch_adj_logits_batch,
            frozen_patch_feature_logits_batch=frozen_patch_feature_logits_batch,
            frozen_patch_adj_logits_batch=frozen_patch_adj_logits_batch,
        )

    def export_full_patch_binary_batch(
        self,
        fixed_attack_adj_binary_batch: Tensor,
        fixed_attack_features_binary_batch: Tensor,
        trigger_feature_logits: Tensor,
        trigger_adj_logits_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            full_adj_logits_batch, full_feature_logits_batch = self.build_full_patch_logits_batch(
                fixed_attack_adj_binary_batch=fixed_attack_adj_binary_batch,
                fixed_attack_features_binary_batch=fixed_attack_features_binary_batch,
                trigger_feature_logits=trigger_feature_logits,
                trigger_adj_logits_rows=trigger_adj_logits_rows,
            )
            full_adj_binary_batch, full_feature_binary_batch = self._hard_binarize_patch_outputs(
                full_adj_logits_batch,
                full_feature_logits_batch,
            )
            return full_adj_binary_batch, full_feature_binary_batch

    def generate_full_patch_batch_for_targets(
        self,
        data,
        target_node_indices: Sequence[int],
        binary: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        full patch 语义：
        - trigger 区域来自当前 generator
        - pure patch 区域来自第一阶段冻结快照
        - 从而保证第二阶段不会改掉第一阶段 patch
        """
        data = data.to(self.device)
        target_node_indices = [int(x) for x in target_node_indices]

        with torch.no_grad():
            full_feature_logits_batch, full_adj_logits_batch = self._compose_full_patch_logits_for_targets(
                data=data,
                target_node_indices=target_node_indices,
            )

            if binary:
                full_adj_binary_batch, full_feature_binary_batch = self._hard_binarize_patch_outputs(
                    full_adj_logits_batch,
                    full_feature_logits_batch,
                )
                return full_feature_binary_batch, full_adj_binary_batch

            return full_feature_logits_batch, full_adj_logits_batch


    def _normalize_target_nodes(self, target_node_indices: Sequence[int]) -> List[int]:
        out = []
        seen = set()
        for item in target_node_indices:
            idx = int(item)
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        if len(out) == 0:
            raise ValueError("target_node_indices 为空。")
        return out



    def _iter_target_node_batches(
            self,
            target_node_indices: Sequence[int],
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ):
        target_nodes = self._normalize_target_nodes(target_node_indices)
        effective_batch_size = self.default_train_batch_size if batch_size is None else int(batch_size)
        effective_batch_size = max(1, min(len(target_nodes), effective_batch_size))

        target_nodes = list(target_nodes)
        if shuffle and len(target_nodes) > 1:
            random.shuffle(target_nodes)

        for start in range(0, len(target_nodes), effective_batch_size):
            yield target_nodes[start:start + effective_batch_size]

    def _apply_structure_keep_mask(
        self,
        patch_adj_tensor_for_structure: Tensor,
        structure_keep_mask: Optional[Tensor] = None,
    ) -> Tensor:
        patch_adj_tensor_for_structure = self._to_hard_adj_tensor(patch_adj_tensor_for_structure)
        if structure_keep_mask is None:
            return patch_adj_tensor_for_structure

        mask = (structure_keep_mask.to(patch_adj_tensor_for_structure.device) >= 0.5).float()
        mask = self._sanitize_binary_adj(mask)
        return self._sanitize_binary_adj(patch_adj_tensor_for_structure * mask)

    def _apply_patch_to_graph(
        self,
        original_features: Tensor,
        original_edge_index: Tensor,
        target_node_idx: int,
        patch_adj_tensor_for_structure: Tensor,
        patch_node_features: Tensor,
        clean_edge_index_override: Optional[Tensor] = None,
        structure_keep_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        patch_adj_tensor_for_structure, patch_node_features = self._validate_patch_tensor_shapes(
            patch_adj_tensor_for_structure,
            patch_node_features,
        )

        base_edge_index = (
            clean_edge_index_override.to(original_features.device)
            if clean_edge_index_override is not None
            else original_edge_index.to(original_features.device)
        )

        num_original_nodes = original_features.shape[0]
        num_pure_patch_nodes = self.num_patch_nodes - self.num_trigger_nodes
        if num_pure_patch_nodes <= 0:
            return original_features, _finalize_binary_undirected_edge_index(base_edge_index, original_features.shape[0])

        pure_patch_features = patch_node_features[self.num_trigger_nodes :]
        pure_patch_features_binary = self._to_hard_feature_tensor(pure_patch_features)
        new_features = torch.cat([original_features, pure_patch_features_binary], dim=0)

        patch_adj_tensor_for_structure = self._apply_structure_keep_mask(
            patch_adj_tensor_for_structure,
            structure_keep_mask=structure_keep_mask,
        )

        new_edge_index_list = [base_edge_index.clone()]

        pure_patch_adj_binary = patch_adj_tensor_for_structure[
            self.num_trigger_nodes : self.num_patch_nodes,
            self.num_trigger_nodes : self.num_patch_nodes,
        ]
        pure_patch_internal_edge_index, _ = dense_to_sparse(pure_patch_adj_binary)
        if pure_patch_internal_edge_index.numel() > 0:
            new_edge_index_list.append(pure_patch_internal_edge_index + num_original_nodes)

        for i in range(num_pure_patch_nodes):
            original_patch_node_idx = self.num_trigger_nodes + i
            patch_node_global_idx = num_original_nodes + i
            if patch_adj_tensor_for_structure[original_patch_node_idx, self.num_patch_nodes] >= 0.5:
                new_edge_index_list.append(
                    torch.tensor(
                        [
                            [patch_node_global_idx, target_node_idx],
                            [target_node_idx, patch_node_global_idx],
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )

        if not new_edge_index_list or all(e.numel() == 0 for e in new_edge_index_list):
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            new_edge_index = torch.cat([e for e in new_edge_index_list if e.numel() > 0], dim=1)

        return new_features, _finalize_binary_undirected_edge_index(new_edge_index, new_features.shape[0])

    def _apply_trigger_to_graph(
        self,
        original_features: Tensor,
        original_edge_index: Tensor,
        target_node_idx: int,
        patch_adj_tensor_for_structure: Tensor,
        patch_node_features: Tensor,
        structure_keep_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        patch_adj_tensor_for_structure, patch_node_features = self._validate_patch_tensor_shapes(
            patch_adj_tensor_for_structure,
            patch_node_features,
        )

        num_nodes_before_trigger = int(original_features.shape[0])
        num_pure_patch_nodes = int(self.num_patch_nodes - self.num_trigger_nodes)

        if self.num_trigger_nodes <= 0:
            return original_features, original_edge_index

        trigger_features = patch_node_features[: self.num_trigger_nodes]
        trigger_features_binary = self._to_hard_feature_tensor(trigger_features)
        new_features = torch.cat([original_features, trigger_features_binary], dim=0)

        patch_adj_tensor_for_structure = self._apply_structure_keep_mask(
            patch_adj_tensor_for_structure,
            structure_keep_mask=structure_keep_mask,
        )

        edge_parts = [original_edge_index]

        trigger_internal_adj_binary = patch_adj_tensor_for_structure[
            : self.num_trigger_nodes,
            : self.num_trigger_nodes,
        ]
        trigger_internal_edge_index, _ = dense_to_sparse(trigger_internal_adj_binary)
        if trigger_internal_edge_index.numel() > 0:
            edge_parts.append(trigger_internal_edge_index + num_nodes_before_trigger)

        trigger_to_target_mask = (
            patch_adj_tensor_for_structure[: self.num_trigger_nodes, self.num_patch_nodes] >= 0.5
        )
        trigger_idx = torch.nonzero(trigger_to_target_mask, as_tuple=False).view(-1)
        if trigger_idx.numel() > 0:
            src = num_nodes_before_trigger + trigger_idx
            dst = torch.full_like(src, int(target_node_idx))
            edge_parts.append(
                torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
            )

        if num_pure_patch_nodes > 0:
            trigger_to_patch_mask = (
                patch_adj_tensor_for_structure[
                    : self.num_trigger_nodes,
                    self.num_trigger_nodes : self.num_patch_nodes,
                ] >= 0.5
            )
            tr_idx, pp_idx = torch.nonzero(trigger_to_patch_mask, as_tuple=True)
            if tr_idx.numel() > 0:
                src = num_nodes_before_trigger + tr_idx
                pure_patch_start = num_nodes_before_trigger - num_pure_patch_nodes
                dst = pure_patch_start + pp_idx
                edge_parts.append(
                    torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
                )

        valid_parts = [e for e in edge_parts if e is not None and e.numel() > 0]
        if len(valid_parts) == 0:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            new_edge_index = torch.cat(valid_parts, dim=1)

        return new_features, _finalize_binary_undirected_edge_index(
            new_edge_index,
            new_features.shape[0],
        )

    def _build_target_region_node_mask(
        self,
        original_edge_index,
        target_node_idx,
        num_total_nodes,
        num_original_nodes,
        num_hops=1,
    ):
        if not hasattr(self, "_region_nodes_cache") or self._region_nodes_cache is None:
            self._region_nodes_cache = {}

        cache_key = (int(target_node_idx), int(num_original_nodes), int(num_hops))

        if cache_key not in self._region_nodes_cache:
            region_nodes, _, _, _ = k_hop_subgraph(
                node_idx=int(target_node_idx),
                num_hops=int(num_hops),
                edge_index=original_edge_index,
                relabel_nodes=False,
                num_nodes=int(num_original_nodes),
            )
            self._region_nodes_cache[cache_key] = region_nodes.detach().cpu()

        region_nodes = self._region_nodes_cache[cache_key].to(original_edge_index.device)

        node_mask = torch.zeros(
            int(num_total_nodes),
            dtype=torch.bool,
            device=original_edge_index.device,
        )
        node_mask[region_nodes] = True

        if int(num_total_nodes) > int(num_original_nodes):
            node_mask[int(num_original_nodes) : int(num_total_nodes)] = True

        return node_mask

    def _random_drop_edges_in_target_region(
        self,
        attacked_edge_index: Tensor,
        original_edge_index: Tensor,
        target_node_idx: int,
        num_total_nodes: int,
        num_original_nodes: int,
        drop_prob: float = 0.99,
        num_hops: int = 1,
    ) -> Tensor:
        if attacked_edge_index is None or attacked_edge_index.numel() == 0:
            return attacked_edge_index

        region_node_mask = self._build_target_region_node_mask(
            original_edge_index=original_edge_index,
            target_node_idx=target_node_idx,
            num_total_nodes=num_total_nodes,
            num_original_nodes=num_original_nodes,
            num_hops=num_hops,
        )

        src = attacked_edge_index[0]
        dst = attacked_edge_index[1]
        local_edge_mask = region_node_mask[src] & region_node_mask[dst]
        self_loop_mask = local_edge_mask & (src == dst)
        droppable_mask = local_edge_mask & (~self_loop_mask)

        if int(droppable_mask.sum().item()) == 0:
            return _finalize_binary_undirected_edge_index(attacked_edge_index, num_total_nodes)

        fixed_edges = attacked_edge_index[:, ~droppable_mask]
        local_edges = attacked_edge_index[:, droppable_mask]

        local_src = local_edges[0]
        local_dst = local_edges[1]

        pair_u = torch.min(local_src, local_dst)
        pair_v = torch.max(local_src, local_dst)
        pair_ids = pair_u * int(num_total_nodes) + pair_v

        unique_pair_ids, inverse_idx = torch.unique(pair_ids, sorted=True, return_inverse=True)
        keep_pair_mask = torch.rand(
            unique_pair_ids.size(0),
            device=attacked_edge_index.device,
        ) > float(drop_prob)
        keep_edge_mask = keep_pair_mask[inverse_idx]

        if int(keep_edge_mask.sum().item()) > 0:
            kept_local_edges = local_edges[:, keep_edge_mask]
            new_edge_index = torch.cat([fixed_edges, kept_local_edges], dim=1)
        else:
            new_edge_index = fixed_edges

        return _finalize_binary_undirected_edge_index(new_edge_index, num_total_nodes)

    def _sample_global_random_defense_edge_indices(
        self,
        original_edge_index: Tensor,
        num_original_nodes: int,
        k_mc: int,
        drop_prob: float,
    ) -> List[Tensor]:
        with torch.no_grad():
            base_edge_index = _finalize_binary_undirected_edge_index(
                original_edge_index.to(self.device),
                int(num_original_nodes),
            )
            unique_edges = self._prepare_unique_undirected_edges(base_edge_index)

            if unique_edges.numel() == 0:
                empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                return [empty_edge_index for _ in range(int(k_mc))]

            num_unique_edges = int(unique_edges.size(1))
            keep_mask = (
                torch.rand(int(k_mc), num_unique_edges, device=self.device) > float(drop_prob)
            )

            defended_edge_indices = []
            for mc_idx in range(int(k_mc)):
                kept_edges = unique_edges[:, keep_mask[mc_idx]]
                if kept_edges.numel() > 0:
                    directed_kept_edges = torch.cat([kept_edges, kept_edges.flip(0)], dim=1)
                else:
                    directed_kept_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)

                defended_edge_indices.append(
                    _finalize_binary_undirected_edge_index(
                        directed_kept_edges,
                        int(num_original_nodes),
                    )
                )
            return defended_edge_indices

    def _sample_patch_structure_keep_masks(self, k_mc: int) -> Tensor:
        with torch.no_grad():
            patch_size_with_target = int(self.num_patch_nodes) + 1
            tri_idx = torch.triu_indices(
                patch_size_with_target,
                patch_size_with_target,
                offset=1,
                device=self.device,
            )
            num_pairs = int(tri_idx.size(1))

            keep_mask = (
                torch.rand(int(k_mc), num_pairs, device=self.device) > float(self.defense_drop_prob)
            ).float()

            structure_masks = torch.zeros(
                int(k_mc),
                patch_size_with_target,
                patch_size_with_target,
                dtype=torch.float32,
                device=self.device,
            )
            structure_masks[:, tri_idx[0], tri_idx[1]] = keep_mask
            structure_masks[:, tri_idx[1], tri_idx[0]] = keep_mask
            structure_masks.diagonal(dim1=-2, dim2=-1).zero_()
            return structure_masks

    def prepare_epoch_defense_cache(
        self,
        data,
        k_mc: Optional[int] = None,
        drop_prob: Optional[float] = None,
    ) -> Dict[str, object]:
        num_mc = int(self.defense_trials if k_mc is None else k_mc)
        num_mc = max(1, num_mc)
        drop_prob = self.defense_drop_prob if drop_prob is None else float(drop_prob)

        clean_edge_indices = self._sample_global_random_defense_edge_indices(
            original_edge_index=data.edge_index,
            num_original_nodes=int(data.x.size(0)),
            k_mc=num_mc,
            drop_prob=drop_prob,
        )
        patch_structure_masks = self._sample_patch_structure_keep_masks(num_mc)

        return {
            "num_mc": int(num_mc),
            "clean_edge_indices": clean_edge_indices,
            "patch_structure_masks": patch_structure_masks,
        }

    def build_full_patch_logits(
        self,
        fixed_attack_adj_binary: Tensor,
        fixed_attack_features_binary: Tensor,
        trigger_feature_logits: Tensor,
        trigger_adj_logits_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        fixed_attack_adj_binary = self._sanitize_binary_adj(
            fixed_attack_adj_binary.to(self.device).float()
        )
        fixed_attack_features_binary = (fixed_attack_features_binary.to(self.device).float() >= 0.5).float()
        trigger_feature_logits = trigger_feature_logits.to(self.device)
        trigger_adj_logits_rows = trigger_adj_logits_rows.to(self.device)

        full_feature_logits = self._binary_to_logits(fixed_attack_features_binary)
        if self.num_trigger_nodes > 0:
            full_feature_logits[: self.num_trigger_nodes] = trigger_feature_logits

        full_adj_logits = self._binary_to_logits(fixed_attack_adj_binary)
        full_adj_logits.diagonal(dim1=-2, dim2=-1).fill_(-self.binary_logit_scale)

        if self.num_trigger_nodes > 0:
            internal_trigger_logits = trigger_adj_logits_rows[:, : self.num_trigger_nodes]
            internal_trigger_logits = 0.5 * (internal_trigger_logits + internal_trigger_logits.t())
            internal_trigger_logits = internal_trigger_logits - torch.diag_embed(
                torch.diagonal(internal_trigger_logits)
            )
            full_adj_logits[: self.num_trigger_nodes, : self.num_trigger_nodes] = internal_trigger_logits

        if self.num_trigger_nodes < self.num_patch_nodes:
            trigger_to_patch_logits = trigger_adj_logits_rows[
                :,
                self.num_trigger_nodes : self.num_patch_nodes,
            ]
            full_adj_logits[: self.num_trigger_nodes, self.num_trigger_nodes : self.num_patch_nodes] = (
                trigger_to_patch_logits
            )
            full_adj_logits[self.num_trigger_nodes : self.num_patch_nodes, : self.num_trigger_nodes] = (
                trigger_to_patch_logits.t()
            )

        if self.num_trigger_nodes > 0:
            trigger_to_target_logits = trigger_adj_logits_rows[:, self.num_patch_nodes]
            full_adj_logits[: self.num_trigger_nodes, self.num_patch_nodes] = trigger_to_target_logits
            full_adj_logits[self.num_patch_nodes, : self.num_trigger_nodes] = trigger_to_target_logits

        full_adj_logits.diagonal(dim1=-2, dim2=-1).fill_(-self.binary_logit_scale)
        return full_adj_logits, full_feature_logits

    def build_full_patch_logits_batch(
        self,
        fixed_attack_adj_binary_batch: Tensor,
        fixed_attack_features_binary_batch: Tensor,
        trigger_feature_logits: Tensor,
        trigger_adj_logits_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        single_input = fixed_attack_adj_binary_batch.dim() == 2
        if single_input:
            fixed_attack_adj_binary_batch = fixed_attack_adj_binary_batch.unsqueeze(0)
            fixed_attack_features_binary_batch = fixed_attack_features_binary_batch.unsqueeze(0)

        batch_size = int(fixed_attack_adj_binary_batch.size(0))
        fixed_attack_adj_binary_batch = self._sanitize_binary_adj(
            fixed_attack_adj_binary_batch.to(self.device).float()
        )
        fixed_attack_features_binary_batch = (
            fixed_attack_features_binary_batch.to(self.device).float() >= 0.5
        ).float()

        trigger_feature_logits = trigger_feature_logits.to(self.device)
        trigger_adj_logits_rows = trigger_adj_logits_rows.to(self.device)

        full_feature_logits = self._binary_to_logits(fixed_attack_features_binary_batch)
        if self.num_trigger_nodes > 0:
            full_feature_logits[:, : self.num_trigger_nodes, :] = (
                trigger_feature_logits.unsqueeze(0).expand(batch_size, -1, -1)
            )

        full_adj_logits = self._binary_to_logits(fixed_attack_adj_binary_batch)
        full_adj_logits.diagonal(dim1=-2, dim2=-1).fill_(-self.binary_logit_scale)

        if self.num_trigger_nodes > 0:
            internal_trigger_logits = trigger_adj_logits_rows[:, : self.num_trigger_nodes]
            internal_trigger_logits = 0.5 * (internal_trigger_logits + internal_trigger_logits.t())
            internal_trigger_logits = internal_trigger_logits - torch.diag_embed(
                torch.diagonal(internal_trigger_logits)
            )
            full_adj_logits[:, : self.num_trigger_nodes, : self.num_trigger_nodes] = (
                internal_trigger_logits.unsqueeze(0).expand(batch_size, -1, -1)
            )

        if self.num_trigger_nodes < self.num_patch_nodes:
            trigger_to_patch_logits = trigger_adj_logits_rows[
                :,
                self.num_trigger_nodes : self.num_patch_nodes,
            ]
            full_adj_logits[:, : self.num_trigger_nodes, self.num_trigger_nodes : self.num_patch_nodes] = (
                trigger_to_patch_logits.unsqueeze(0).expand(batch_size, -1, -1)
            )
            full_adj_logits[:, self.num_trigger_nodes : self.num_patch_nodes, : self.num_trigger_nodes] = (
                trigger_to_patch_logits.t().unsqueeze(0).expand(batch_size, -1, -1)
            )

        if self.num_trigger_nodes > 0:
            trigger_to_target_logits = trigger_adj_logits_rows[:, self.num_patch_nodes]
            full_adj_logits[:, : self.num_trigger_nodes, self.num_patch_nodes] = (
                trigger_to_target_logits.unsqueeze(0).expand(batch_size, -1)
            )
            full_adj_logits[:, self.num_patch_nodes, : self.num_trigger_nodes] = (
                trigger_to_target_logits.unsqueeze(0).expand(batch_size, -1)
            )

        full_adj_logits.diagonal(dim1=-2, dim2=-1).fill_(-self.binary_logit_scale)

        if single_input:
            return full_adj_logits.squeeze(0), full_feature_logits.squeeze(0)
        return full_adj_logits, full_feature_logits

    def export_full_patch_binary(
        self,
        fixed_attack_adj_binary: Tensor,
        fixed_attack_features_binary: Tensor,
        trigger_feature_logits: Tensor,
        trigger_adj_logits_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            full_adj_logits, full_feature_logits = self.build_full_patch_logits(
                fixed_attack_adj_binary=fixed_attack_adj_binary,
                fixed_attack_features_binary=fixed_attack_features_binary,
                trigger_feature_logits=trigger_feature_logits,
                trigger_adj_logits_rows=trigger_adj_logits_rows,
            )
            return self._hard_binarize_patch_outputs(full_adj_logits, full_feature_logits)

    def create_trigger_state(
        self,
        init_patch_adj_logits: Tensor,
        init_patch_feature_logits: Tensor,
        lr: float = 2e-5,
        weight_decay: float = 1e-5,
    ) -> Dict[str, object]:
        if self.num_trigger_nodes <= 0:
            raise RuntimeError("num_trigger_nodes <= 0，不能创建 trigger state。")

        trigger_feature_logits = nn.Parameter(
            init_patch_feature_logits[: self.num_trigger_nodes].detach().clone().to(self.device)
        )
        trigger_adj_logits_rows = nn.Parameter(
            init_patch_adj_logits[: self.num_trigger_nodes, :].detach().clone().to(self.device)
        )
        optimizer = optim.AdamW(
            [trigger_feature_logits, trigger_adj_logits_rows],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        return {
            "trigger_feature_logits": trigger_feature_logits,
            "trigger_adj_logits_rows": trigger_adj_logits_rows,
            "optimizer": optimizer,
        }

    def generate_full_patch_for_target(
            self,
            data,
            target_node_idx: int,
            binary: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        feature_batch, adj_batch = self.generate_full_patch_batch_for_targets(
            data=data,
            target_node_indices=[int(target_node_idx)],
            binary=binary,
        )
        return feature_batch.squeeze(0), adj_batch.squeeze(0)

    def _build_patch_only_data_list(
        self,
        data,
        target_nodes: Sequence[int],
        patch_adj_batch: Tensor,
        patch_feature_batch: Tensor,
        clean_edge_index: Tensor,
        structure_keep_mask: Tensor,
    ) -> List[Data]:
        graph_list = []
        for i, node_idx in enumerate(target_nodes):
            attacked_x, attacked_edge_index = self._apply_patch_to_graph(
                original_features=data.x,
                original_edge_index=data.edge_index,
                target_node_idx=int(node_idx),
                patch_adj_tensor_for_structure=patch_adj_batch[i],
                patch_node_features=patch_feature_batch[i],
                clean_edge_index_override=clean_edge_index,
                structure_keep_mask=structure_keep_mask,
            )
            graph_data = Data(x=attacked_x, edge_index=attacked_edge_index)
            graph_data.target_node_idx = int(node_idx)
            graph_list.append(graph_data)
        return graph_list

    def _build_full_patch_data_list(
        self,
        data,
        target_nodes: Sequence[int],
        full_patch_adj_batch: Tensor,
        full_patch_feature_batch: Tensor,
        clean_edge_index: Tensor,
        structure_keep_mask: Tensor,
    ) -> List[Data]:
        graph_list = []
        for i, node_idx in enumerate(target_nodes):
            attacked_x, attacked_edge_index = self._apply_patch_to_graph(
                original_features=data.x,
                original_edge_index=data.edge_index,
                target_node_idx=int(node_idx),
                patch_adj_tensor_for_structure=full_patch_adj_batch[i],
                patch_node_features=full_patch_feature_batch[i],
                clean_edge_index_override=clean_edge_index,
                structure_keep_mask=structure_keep_mask,
            )

            if self.num_trigger_nodes > 0:
                attacked_x, attacked_edge_index = self._apply_trigger_to_graph(
                    original_features=attacked_x,
                    original_edge_index=attacked_edge_index,
                    target_node_idx=int(node_idx),
                    patch_adj_tensor_for_structure=full_patch_adj_batch[i],
                    patch_node_features=full_patch_feature_batch[i],
                    structure_keep_mask=structure_keep_mask,
                )

            graph_data = Data(x=attacked_x, edge_index=attacked_edge_index)
            graph_data.target_node_idx = int(node_idx)
            graph_list.append(graph_data)
        return graph_list

    def _batch_graph_list_and_collect_target_indices(
        self,
        graph_list: List[Data],
    ) -> Tuple[Batch, Tensor]:
        batched_graph = Batch.from_data_list(graph_list).to(self.device)
        local_target_indices = torch.tensor(
            [int(graph.target_node_idx) for graph in graph_list],
            dtype=torch.long,
            device=self.device,
        )
        global_target_indices = batched_graph.ptr[:-1].to(self.device) + local_target_indices
        return batched_graph, global_target_indices

    def train_patch_shared(
        self,
        data,
        target_node_indices: Sequence[int],
        base_epochs: int = 1,
        lambda_attack: float = 1.0,
        attack_target_class: int = 0,
        k_mc: Optional[int] = None,
        clip_grad_norm: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        if self.num_patch_nodes - self.num_trigger_nodes <= 0:
            raise ValueError("num_patch_nodes - num_trigger_nodes 必须大于 0，否则没有 pure patch 可训练。")

        data = data.to(self.device)
        target_nodes = self._normalize_target_nodes(target_node_indices)
        self.clean_gat_model.eval()
        self.patch_generator.train()

        num_mc = int(self.defense_trials if k_mc is None else k_mc)
        num_mc = max(1, num_mc)

        best_state = None
        best_epoch = 0
        best_loss = float("inf")
        best_target_rate = -1.0
        best_misclf_rate = -1.0
        best_attack_success_code = 0

        effective_batch_size = min(
            len(target_nodes),
            self.default_train_batch_size if batch_size is None else int(batch_size),
        )
        effective_batch_size = max(1, effective_batch_size)

        for epoch in range(1, max(1, int(base_epochs)) + 1):
            epoch_cache = self.prepare_epoch_defense_cache(
                data=data,
                k_mc=num_mc,
                drop_prob=self.defense_drop_prob,
            )

            epoch_loss_sum = 0.0
            epoch_target_hits = 0
            epoch_misclf_hits = 0
            epoch_total_cases = 0

            for batch_nodes in self._iter_target_node_batches(
                    target_node_indices=target_nodes,
                    batch_size=effective_batch_size,
                    shuffle=True,
            ):
                batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=self.device)
                true_labels = data.y[batch_nodes_tensor]
                attack_target_labels = torch.full(
                    (len(batch_nodes),),
                    int(attack_target_class),
                    dtype=torch.long,
                    device=self.device,
                )

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    patch_feature_logits_batch, patch_adj_logits_batch = self.generate_patch_batch_for_targets(
                        data=data,
                        target_node_indices=batch_nodes,
                    )

                    batch_loss = torch.tensor(0.0, device=self.device)
                    batch_target_hits = 0
                    batch_misclf_hits = 0

                    for mc_idx in range(num_mc):
                        clean_edge_index_mc = epoch_cache["clean_edge_indices"][mc_idx]
                        structure_keep_mask_mc = epoch_cache["patch_structure_masks"][mc_idx]

                        graph_list = self._build_patch_only_data_list(
                            data=data,
                            target_nodes=batch_nodes,
                            patch_adj_batch=patch_adj_logits_batch,
                            patch_feature_batch=patch_feature_logits_batch,
                            clean_edge_index=clean_edge_index_mc,
                            structure_keep_mask=structure_keep_mask_mc,
                        )
                        batched_graph, global_target_indices = self._batch_graph_list_and_collect_target_indices(
                            graph_list
                        )

                        out = _forward_clean_model(
                            self.clean_gat_model,
                            batched_graph.x,
                            batched_graph.edge_index,
                        )
                        pred_target = out[global_target_indices]

                        mc_loss = (
                                float(lambda_attack)
                                * F.nll_loss(pred_target, attack_target_labels, reduction="mean")
                                / float(num_mc)
                        )
                        batch_loss += mc_loss

                        pred_class = pred_target.argmax(dim=1)
                        batch_target_hits += int((pred_class == attack_target_labels).sum().item())
                        batch_misclf_hits += int((pred_class != true_labels).sum().item())

                self._backward_step_patch_generator(batch_loss, clip_grad_norm)

                epoch_loss_sum += float(batch_loss.item()) * len(batch_nodes)
                epoch_target_hits += batch_target_hits
                epoch_misclf_hits += batch_misclf_hits
                epoch_total_cases += len(batch_nodes) * num_mc

            epoch_loss = epoch_loss_sum / float(len(target_nodes))
            epoch_target_rate = epoch_target_hits / float(max(1, epoch_total_cases))
            epoch_misclf_rate = epoch_misclf_hits / float(max(1, epoch_total_cases))
            if epoch_target_rate >= 0.5:
                epoch_attack_success_code = 2
            elif epoch_misclf_rate >= 0.5:
                epoch_attack_success_code = 1
            else:
                epoch_attack_success_code = 0

            if (
                epoch_target_rate > best_target_rate
                or (
                    abs(epoch_target_rate - best_target_rate) <= 1e-12
                    and epoch_loss < best_loss
                )
            ):
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.patch_generator.state_dict().items()
                }
                best_epoch = int(epoch)
                best_loss = float(epoch_loss)
                best_target_rate = float(epoch_target_rate)
                best_misclf_rate = float(epoch_misclf_rate)
                best_attack_success_code = int(epoch_attack_success_code)

            if epoch == 1 or epoch % 5 == 0 or epoch == int(base_epochs):
                print(
                    f"[SharedPatch-BatchMC] epoch {epoch}/{base_epochs} "
                    f"batch={len(batch_nodes)} "
                    f"mc={num_mc} "
                    f"loss={epoch_loss:.4f} "
                    f"target_rate={epoch_target_rate:.4f} "
                    f"misclf_rate={epoch_misclf_rate:.4f}"
                )

        if best_state is not None:
            self.patch_generator.load_state_dict(best_state)

        self.patch_generator.eval()
        self._refresh_stage1_frozen_patch_generator()

        seed_node_idx = int(target_nodes[0])

        with torch.no_grad():
            seed_patch_features_logits, seed_patch_adj_logits = self.generate_patch_for_target(
                data=data,
                target_node_idx=seed_node_idx,
            )
            self.best_patch_adj_logits = seed_patch_adj_logits.detach().clone()
            self.best_patch_node_features_logits = seed_patch_features_logits.detach().clone()
            self.best_patch_adj, self.best_patch_node_features = self._hard_binarize_patch_outputs(
                self.best_patch_adj_logits,
                self.best_patch_node_features_logits,
            )

        return {
            "mode": "shared_generator_batched_mc_cache",
            "generator_subgraph_hops": int(self.generator_subgraph_hops),
            "train_node_count": int(len(target_nodes)),
            "epochs": int(base_epochs),
            "effective_batch_size": int(effective_batch_size),
            "mc_cache_size": int(num_mc),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "best_target_success_rate": float(best_target_rate),
            "best_misclassification_rate": float(best_misclf_rate),
            "best_attack_success_code": int(best_attack_success_code),
            "seed_node_idx": int(seed_node_idx),
        }

    def train_trigger_shared(
            self,
            data,
            target_node_indices: Sequence[int],
            base_epochs: int = 1,
            attack_target_class: int = 0,
            k_mc: Optional[int] = None,
            lambda_recover: float = 1.0,
            lambda_margin: float = 0.5,
            clip_grad_norm: float = 1.0,
            margin: float = 0.2,
            batch_size: Optional[int] = None,
            trigger_lr: float = 2e-5,
            trigger_weight_decay: float = 1e-5,
    ) -> Dict[str, object]:
        """
        新语义：
        - trigger 不再是共享参数
        - trigger 和 patch 一样，都是共享 generator 对每个目标节点独立生成
        - 第二阶段训练实际上是在 full patch 目标下继续微调同一个 generator
        """
        if self.num_trigger_nodes <= 0:
            return {
                "mode": "per_node_trigger_disabled",
                "skipped": True,
                "reason": "num_trigger_nodes <= 0",
            }

        data = data.to(self.device)
        target_nodes = self._normalize_target_nodes(target_node_indices)
        self.clean_gat_model.eval()
        self.patch_generator.train()
        if self.freeze_patch_during_trigger_finetune:
            self._ensure_stage1_frozen_patch_generator()

        if float(trigger_lr) > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = float(trigger_lr)
                group["weight_decay"] = float(trigger_weight_decay)

        num_mc = int(self.defense_trials if k_mc is None else k_mc)
        num_mc = max(1, num_mc)

        best_state = None
        best_epoch = 0
        best_loss = float("inf")
        best_recovery_rate = -1.0
        best_suppression_rate = -1.0

        effective_batch_size = min(
            len(target_nodes),
            self.default_train_batch_size if batch_size is None else int(batch_size),
        )
        effective_batch_size = max(1, effective_batch_size)

        for epoch in range(1, max(1, int(base_epochs)) + 1):
            epoch_cache = self.prepare_epoch_defense_cache(
                data=data,
                k_mc=num_mc,
                drop_prob=self.defense_drop_prob,
            )

            epoch_loss_sum = 0.0
            epoch_recovery_hits = 0
            epoch_suppression_hits = 0
            epoch_total_cases = 0

            for batch_nodes in self._iter_target_node_batches(
                    target_node_indices=target_nodes,
                    batch_size=effective_batch_size,
                    shuffle=True,
            ):
                batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=self.device)
                true_labels = data.y[batch_nodes_tensor]

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    # 第二阶段：trigger 用当前 generator，patch 用第一阶段冻结快照
                    full_feature_logits_batch, full_adj_logits_batch = self._compose_full_patch_logits_for_targets(
                        data=data,
                        target_node_indices=batch_nodes,
                    )


                    batch_loss = torch.tensor(0.0, device=self.device)
                    batch_recovery_hits = 0
                    batch_suppression_hits = 0

                    for mc_idx in range(num_mc):
                        clean_edge_index_mc = epoch_cache["clean_edge_indices"][mc_idx]
                        structure_keep_mask_mc = epoch_cache["patch_structure_masks"][mc_idx]

                        graph_list = self._build_full_patch_data_list(
                            data=data,
                            target_nodes=batch_nodes,
                            full_patch_adj_batch=full_adj_logits_batch,
                            full_patch_feature_batch=full_feature_logits_batch,
                            clean_edge_index=clean_edge_index_mc,
                            structure_keep_mask=structure_keep_mask_mc,
                        )
                        batched_graph, global_target_indices = self._batch_graph_list_and_collect_target_indices(
                            graph_list
                        )

                        out = _forward_clean_model(
                            self.clean_gat_model,
                            batched_graph.x,
                            batched_graph.edge_index,
                        )
                        pred_target = out[global_target_indices]

                        true_logp = pred_target.gather(1, true_labels.view(-1, 1)).squeeze(1)
                        target_logp = pred_target[:, int(attack_target_class)]

                        loss_recover = F.nll_loss(pred_target, true_labels, reduction="mean")
                        loss_margin = F.relu(target_logp - true_logp + float(margin)).mean()

                        mc_loss = (
                                          float(lambda_recover) * loss_recover
                                          + float(lambda_margin) * loss_margin
                                  ) / float(num_mc)

                        batch_loss += mc_loss

                        pred_class = pred_target.argmax(dim=1)
                        batch_recovery_hits += int((pred_class == true_labels).sum().item())
                        batch_suppression_hits += int((pred_class != int(attack_target_class)).sum().item())

                self._backward_step_patch_generator(batch_loss, clip_grad_norm)

                epoch_loss_sum += float(batch_loss.item()) * len(batch_nodes)
                epoch_recovery_hits += batch_recovery_hits
                epoch_suppression_hits += batch_suppression_hits
                epoch_total_cases += len(batch_nodes) * num_mc

            epoch_loss = epoch_loss_sum / float(len(target_nodes))
            epoch_recovery_rate = epoch_recovery_hits / float(max(1, epoch_total_cases))
            epoch_suppression_rate = epoch_suppression_hits / float(max(1, epoch_total_cases))

            if (
                    epoch_recovery_rate > best_recovery_rate
                    or (
                    abs(epoch_recovery_rate - best_recovery_rate) <= 1e-12
                    and epoch_suppression_rate > best_suppression_rate
            )
                    or (
                    abs(epoch_recovery_rate - best_recovery_rate) <= 1e-12
                    and abs(epoch_suppression_rate - best_suppression_rate) <= 1e-12
                    and epoch_loss < best_loss
            )
            ):
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.patch_generator.state_dict().items()
                }
                best_epoch = int(epoch)
                best_loss = float(epoch_loss)
                best_recovery_rate = float(epoch_recovery_rate)
                best_suppression_rate = float(epoch_suppression_rate)

            if epoch == 1 or epoch % 5 == 0 or epoch == int(base_epochs):
                print(
                    f"[PerNodeTrigger-BatchMC] epoch {epoch}/{base_epochs} "
                    f"batch={effective_batch_size} "
                    f"mc={num_mc} "
                    f"loss={epoch_loss:.4f} "
                    f"recovery_rate={epoch_recovery_rate:.4f} "
                    f"suppression_rate={epoch_suppression_rate:.4f}"
                )

        if best_state is not None:
            self.patch_generator.load_state_dict(best_state)

        self.patch_generator.eval()

        # 保留一个 seed 节点的 full patch 仅作为缓存/兼容旧评估接口
        seed_node_idx = int(target_nodes[0])
        with torch.no_grad():
            seed_full_feature_logits, seed_full_adj_logits = self.generate_full_patch_for_target(
                data=data,
                target_node_idx=seed_node_idx,
                binary=False,
            )
            self.best_full_patch_adj_logits = seed_full_adj_logits.detach().clone()
            self.best_full_patch_node_features_logits = seed_full_feature_logits.detach().clone()
            self.best_full_patch_adj, self.best_full_patch_node_features = self._hard_binarize_patch_outputs(
                self.best_full_patch_adj_logits,
                self.best_full_patch_node_features_logits,
            )

        # 兼容旧字段，但这里不再表示共享 trigger 参数
        self.best_trigger_feature_logits = None
        self.best_trigger_adj_logits_rows = None

        return {
            "mode": "per_node_trigger_via_shared_generator",
            "generator_subgraph_hops": int(self.generator_subgraph_hops),
            "train_node_count": int(len(target_nodes)),
            "epochs": int(base_epochs),
            "effective_batch_size": int(effective_batch_size),
            "mc_cache_size": int(num_mc),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "best_recovery_rate": float(best_recovery_rate),
            "best_attack_suppression_rate": float(best_suppression_rate),
            "patch_region_frozen_by_mask": bool(self.freeze_patch_during_trigger_finetune),
            "seed_node_idx": int(seed_node_idx),
        }

    def train_patch(
        self,
        data,
        target_node_original_idx: int,
        target_label: Tensor,
        base_epochs: int = 1,
        lambda_attack: float = 1.0,
        attack_target_class: int = 0,
        k_mc: Optional[int] = None,
        clip_grad_norm: float = 1.0,
    ) -> Tuple[Tensor, Tensor, int]:
        metrics = self.train_patch_shared(
            data=data,
            target_node_indices=[int(target_node_original_idx)],
            base_epochs=base_epochs,
            lambda_attack=lambda_attack,
            attack_target_class=attack_target_class,
            k_mc=k_mc,
            clip_grad_norm=clip_grad_norm,
            batch_size=1,
        )
        data = data.to(self.device)
        with torch.no_grad():
            patch_node_features_logits, patch_adj_logits = self.generate_patch_for_target(
                data=data,
                target_node_idx=int(target_node_original_idx),
            )
        return (
            patch_adj_logits.detach().clone(),
            patch_node_features_logits.detach().clone(),
            int(metrics.get("best_attack_success_code", 0)),
        )

    def train_trigger(
        self,
        data,
        target_node_original_idx: int,
        target_label: Tensor,
        fixed_attack_adj_binary: Tensor,
        fixed_attack_features_binary: Tensor,
        trigger_state: Dict[str, object],
        base_epochs: int = 1,
        attack_target_class: int = 0,
        k_mc: Optional[int] = None,
        lambda_recover: float = 1.0,
        lambda_margin: float = 0.5,
        clip_grad_norm: float = 1.0,
        margin: float = 0.2,
    ) -> Tuple[Tensor, Tensor, int]:
        metrics = self.train_trigger_shared(
            data=data,
            target_node_indices=[int(target_node_original_idx)],
            base_epochs=base_epochs,
            attack_target_class=attack_target_class,
            k_mc=k_mc,
            lambda_recover=lambda_recover,
            lambda_margin=lambda_margin,
            clip_grad_norm=clip_grad_norm,
            margin=margin,
            batch_size=1,
        )
        data = data.to(self.device)
        with torch.no_grad():
            full_feature_logits, full_adj_logits = self.generate_full_patch_for_target(
                data=data,
                target_node_idx=int(target_node_original_idx),
                binary=False,
            )
        attack_success = 1 if float(metrics.get("best_recovery_rate", 0.0)) > 0.5 else 0
        return (
            full_adj_logits.detach().clone(),
            full_feature_logits.detach().clone(),
            int(attack_success),
        )


PatchTrainer = PatchTrainerGNN
PatchTrainerGNN_DNN_1 = PatchTrainerGNN
PatchTrainerGNN_DNN_1_updated = PatchTrainerGNN
