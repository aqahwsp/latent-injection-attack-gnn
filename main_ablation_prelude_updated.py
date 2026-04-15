# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import inspect
import json
import os
import random
import traceback
import hashlib
from pathlib import Path
import gc
def cleanup_trainer(trainer):
    if trainer is not None:
        try:
            trainer.release_unused_cache(keep_best=False)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

import numpy as np
import torch
import torch.nn.functional as F

import preprocess
import models
import attack_evaluation_updated
import PatchTrainerGNN_DNN_1_updated as patch_trainer_module
from utils import cleanup_memory
import gc
import train_clean_gat as train_clean_gat_module
import train_clean_gcn as train_clean_gcn_module


TRAIN_CLEAN_MODULES = {
    "GAT": train_clean_gat_module,
    "GCN": train_clean_gcn_module,
}

try:
    from torch_geometric.data import Data
    from torch_geometric.datasets import Flickr, Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.utils import coalesce, remove_self_loops, to_undirected
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "当前环境缺少 torch_geometric，请先安装 PyG 后再运行消融实验。"
    ) from exc



DATASET_NAME_ALIASES = {
    "Citeseer": "CiteSeer",
    "citeseer": "CiteSeer",
    "Pubmed": "PubMed",
    "pubmed": "PubMed",
}

DATASET_ORDER = ["PubMed", "Cora", "CiteSeer", "Flickr"]

BACKBONE_NAME_ALIASES = {
    "gat": "GAT",
    "GCAT": "GAT",
    "gcn": "GCN",
}

BACKBONE_ORDER = ["GAT", "GCN"]

DATASET_NPZ_BASENAMES = {
    "Cora": ["cora.npz", "Cora.npz"],
    "CiteSeer": ["citeseer.npz", "CiteSeer.npz"],
    "PubMed": ["pubmed.npz", "PubMed.npz"],
    "Flickr": ["flickr.npz", "Flickr.npz"],
}

DEFAULT_TRIGGER_BASE_RATIO = 0.5
DEFAULT_TRIGGER_RATIO_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7]


DEFAULT_TRIGGER_TRAIN_EPOCHS = 60
DEFAULT_TRAIN_DEFENSE_TRIALS = 32
DEFAULT_EVAL_DEFENSE_TRIALS = 1000
DEFAULT_MAX_EVALUATION_NODES = None
SAFE_MAX_PATCH_TRAIN_BATCH_SIZE = 256

PRELUDE_BACKBONE = "GAT"
PRELUDE_DATASETS = ["PubMed", "Cora", "CiteSeer", "Flickr"]
PRELUDE_REPEAT_COUNT = 1
PRELUDE_FIXED_PATCH_NODES = 60
PRELUDE_FIXED_TRIGGER_RATIO = 0.5
PRELUDE_EVAL_DEFENSE_TRIALS = 1000


def _make_dataset_ablation_config(max_evaluation_nodes=DEFAULT_MAX_EVALUATION_NODES):
    return {
        "patch_nodes_list": [40, 60, 80, 100, 150, 200],
        "trigger_base_ratio": DEFAULT_TRIGGER_BASE_RATIO,
        "trigger_ratio_levels": DEFAULT_TRIGGER_RATIO_LEVELS,
        "patch_train_epochs": 50,
        "trigger_train_epochs": DEFAULT_TRIGGER_TRAIN_EPOCHS,
        "target_region_hops": 1,
        "generator_subgraph_hops": 1,
        "defense_drop_prob": 0.99,
        "defense_trials": DEFAULT_TRAIN_DEFENSE_TRIALS,
        "train_defense_trials": DEFAULT_TRAIN_DEFENSE_TRIALS,
        "eval_defense_trials": DEFAULT_EVAL_DEFENSE_TRIALS,
        "max_evaluation_nodes": max_evaluation_nodes,
        "attack_target_class": 0,
        "generator_d_model": 192,
        "generator_nhead": 8,
        "generator_num_decoder_layers": 4,
        "generator_dim_feedforward": 512,
        "train_batch_size": 16,
    }




DATASET_ABLATION_CONFIG = {
    "Cora": _make_dataset_ablation_config(),
    "CiteSeer": _make_dataset_ablation_config(),
    "PubMed": _make_dataset_ablation_config(),
    "Flickr": _make_dataset_ablation_config(max_evaluation_nodes=512),
    "_default": _make_dataset_ablation_config(),
}


def compute_num_trigger_nodes_from_ratio(num_patch_nodes, trigger_ratio):
    num_patch_nodes = int(num_patch_nodes)
    if num_patch_nodes <= 0:
        raise ValueError("num_patch_nodes 必须为正整数。")
    trigger_nodes = int(round(num_patch_nodes * float(trigger_ratio)))
    trigger_nodes = max(1, min(num_patch_nodes, trigger_nodes))
    return trigger_nodes


def compute_default_num_trigger_nodes(num_patch_nodes, base_ratio=DEFAULT_TRIGGER_BASE_RATIO):
    return compute_num_trigger_nodes_from_ratio(num_patch_nodes, base_ratio)


def build_num_trigger_nodes_candidates(num_patch_nodes, exp_cfg):
    num_patch_nodes = int(num_patch_nodes)
    ratio_levels = exp_cfg.get("trigger_ratio_levels", DEFAULT_TRIGGER_RATIO_LEVELS)

    # 现在 trigger_ratio_levels 直接解释为 trigger_ratio，
    # 不再乘 trigger_base_ratio。
    if ratio_levels is None:
        ratio_levels = [float(exp_cfg.get("trigger_base_ratio", DEFAULT_TRIGGER_BASE_RATIO))]
    elif isinstance(ratio_levels, (int, float)):
        ratio_levels = [float(ratio_levels)]
    else:
        ratio_levels = [float(item) for item in list(ratio_levels)]

    if len(ratio_levels) == 0:
        ratio_levels = [float(exp_cfg.get("trigger_base_ratio", DEFAULT_TRIGGER_BASE_RATIO))]

    candidates = []
    for ratio in ratio_levels:
        configured_trigger_ratio = float(ratio)
        num_trigger_nodes = compute_num_trigger_nodes_from_ratio(
            num_patch_nodes=num_patch_nodes,
            trigger_ratio=configured_trigger_ratio,
        )
        candidates.append(
            {
                "configured_trigger_ratio": configured_trigger_ratio,
                "num_trigger_nodes": int(num_trigger_nodes),
                "actual_trigger_ratio": float(num_trigger_nodes) / float(num_patch_nodes),
            }
        )

    return candidates


def normalize_dataset_name(name):
    return DATASET_NAME_ALIASES.get(name, name)


def normalize_backbone_name(name):
    if name is None:
        return "GAT"
    key = str(name)
    normalized = BACKBONE_NAME_ALIASES.get(key, BACKBONE_NAME_ALIASES.get(key.lower(), key.upper()))
    if normalized not in BACKBONE_ORDER:
        raise ValueError(f"暂不支持的分类模型类型: {name}。当前仅支持: {', '.join(BACKBONE_ORDER)}")
    return normalized


def normalize_backbone_names(backbones):
    if backbones is None:
        return list(BACKBONE_ORDER)

    normalized = []
    seen = set()
    for item in backbones:
        name = normalize_backbone_name(item)
        if name not in seen:
            normalized.append(name)
            seen.add(name)

    if not normalized:
        return list(BACKBONE_ORDER)
    return normalized


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_first_attr(obj, names, callable_only=False):
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable_only and (not callable(value)):
                continue
            return value
    return None


def filter_kwargs(func, kwargs):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    has_var_kwargs = False
    valid_names = []
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            break
        valid_names.append(name)

    if has_var_kwargs:
        return kwargs

    out = {}
    for key, value in kwargs.items():
        if key in valid_names:
            out[key] = value
    return out


def is_data_like(obj):
    return hasattr(obj, "x") and hasattr(obj, "edge_index")


def extract_data_from_result(result):
    if is_data_like(result):
        return result

    if hasattr(result, "data") and is_data_like(result.data):
        return result.data

    if isinstance(result, dict):
        for key in ["data", "graph", "dataset_obj", "pyg_data"]:
            if key in result and is_data_like(result[key]):
                return result[key]

    if isinstance(result, (list, tuple)):
        for item in result:
            if is_data_like(item):
                return item
            if hasattr(item, "data") and is_data_like(item.data):
                return item.data

    raise RuntimeError("无法从返回值中解析出 PyG Data 对象。")


def ensure_masks(
    data,
    seed=0,
    train_ratio=0.7,
    val_ratio=0,
    test_ratio=0.3,
):
    has_train = hasattr(data, "train_mask") and data.train_mask is not None
    has_val = hasattr(data, "val_mask") and data.val_mask is not None
    has_test = hasattr(data, "test_mask") and data.test_mask is not None

    if has_train and has_val and has_test:
        return data

    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {total_ratio}。")

    num_nodes = int(data.x.size(0))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=generator)

    num_train = int(num_nodes * float(train_ratio))
    num_val = int(num_nodes * float(val_ratio))
    num_test = num_nodes - num_train - num_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train + num_val]] = True
    test_mask[perm[num_train + num_val:num_train + num_val + num_test]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data



def _to_undirected_edge_index_only(edge_index, num_nodes):
    undirected_result = to_undirected(edge_index, num_nodes=int(num_nodes))
    if isinstance(undirected_result, tuple):
        edge_index = undirected_result[0]
    else:
        edge_index = undirected_result
    return edge_index.long()


def _coalesce_edge_index_only(edge_index, num_nodes):
    coalesced_result = coalesce(edge_index, None, int(num_nodes), int(num_nodes))
    if isinstance(coalesced_result, tuple):
        edge_index = coalesced_result[0]
    else:
        edge_index = coalesced_result
    return edge_index.long()


def finalize_edge_index_binary_undirected(edge_index, num_nodes):
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0), dtype=torch.long)
    edge_index = edge_index.long()
    edge_index = _to_undirected_edge_index_only(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = _coalesce_edge_index_only(edge_index, num_nodes)
    return edge_index.long()


def infer_num_classes(data):
    if hasattr(data, "num_classes") and data.num_classes is not None:
        try:
            return int(data.num_classes)
        except Exception:
            pass
    if hasattr(data, "y") and data.y is not None:
        return int(data.y.max().item()) + 1
    raise RuntimeError("无法推断类别数。")


def enrich_data_object(data, dataset_name, source_name, force_resplit=False, split_seed=0):
    data.x = data.x.float()
    if hasattr(preprocess, "normalize_features"):
        data = preprocess.normalize_features(data)
    else:
        data = NormalizeFeatures()(data)

    data.y = data.y.long()
    data.edge_index = finalize_edge_index_binary_undirected(data.edge_index, data.x.size(0))

    if force_resplit:
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None

    data = ensure_masks(data, seed=split_seed)
    data.num_nodes = int(data.x.size(0))
    data.num_features = int(data.x.size(1))
    data.num_classes = int(infer_num_classes(data))
    data.dataset_name = str(dataset_name)
    data.dataset_source = str(source_name)

    return data



def _reference_load_data(npz_path):
    if hasattr(preprocess, "load_data"):
        loaded = preprocess.load_data(npz_path)
        if len(loaded) >= 7:
            return loaded
        raise RuntimeError("preprocess.load_data 返回值格式不正确。")

    with np.load(npz_path, allow_pickle=True) as graph_file:
        graph = {k: np.array(graph_file[k], copy=True) for k in graph_file.files}

    import scipy.sparse as sp

    A = sp.csr_matrix((np.ones(graph["A"].shape[1]).astype(int), graph["A"]))
    X = sp.csr_matrix((np.ones(graph["X"].shape[1]), graph["X"]), dtype=np.float32).todense()
    y = graph["y"]
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc, graph


def _mask_from_graph_value(value, num_nodes):
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or arr.dtype == bool:
        arr = arr.reshape(-1)
        if arr.size == num_nodes:
            return torch.as_tensor(arr, dtype=torch.bool)

    arr = arr.reshape(-1)
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    if arr.size > 0:
        idx = torch.as_tensor(arr, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < int(num_nodes))]
        mask[idx] = True
    return mask


def _extract_masks_from_graph(graph, num_nodes):
    candidate_triples = [
        ("train_mask", "val_mask", "test_mask"),
        ("train_masks", "val_masks", "test_masks"),
        ("idx_train", "idx_val", "idx_test"),
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
    ]
    if graph is None:
        return None, None, None

    for train_key, val_key, test_key in candidate_triples:
        if train_key in graph and val_key in graph and test_key in graph:
            train_mask = _mask_from_graph_value(graph[train_key], num_nodes)
            val_mask = _mask_from_graph_value(graph[val_key], num_nodes)
            test_mask = _mask_from_graph_value(graph[test_key], num_nodes)
            return train_mask, val_mask, test_mask

    return None, None, None


def find_npz_path(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    basenames = DATASET_NPZ_BASENAMES.get(dataset_name, [dataset_name.lower() + ".npz"])
    script_dir = Path(__file__).resolve().parent

    search_dirs = [
        script_dir,
        script_dir / "data",
        script_dir / "DATA",
        script_dir / "tmp",
        script_dir.parent,
        script_dir.parent / "data",
        script_dir.parent / "DATA",
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "DATA",
        Path.cwd() / "tmp",
    ]

    seen = set()
    ordered_dirs = []
    for directory in search_dirs:
        directory = directory.resolve()
        if str(directory) not in seen:
            seen.add(str(directory))
            ordered_dirs.append(directory)

    for directory in ordered_dirs:
        for basename in basenames:
            candidate = directory / basename
            if candidate.exists():
                return candidate
    return None


def load_dataset_from_npz(dataset_name, npz_path):
    loaded = _reference_load_data(str(npz_path))
    if len(loaded) < 7:
        raise RuntimeError("参考 load_data 返回值长度不足，预期至少为 7。")

    A, X, y, n, d, nc, graph = loaded[:7]

    csr_to_edge_index = pick_first_attr(preprocess, ["csr_to_edge_index"], callable_only=True)
    if csr_to_edge_index is not None:
        edge_index = csr_to_edge_index(A)
    else:
        row, col = A.nonzero()
        edge_index = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)

    data = Data(
        x=torch.as_tensor(np.asarray(X), dtype=torch.float32),
        y=torch.as_tensor(np.asarray(y), dtype=torch.long),
        edge_index=edge_index,
        num_nodes=int(n),
    )

    train_mask, val_mask, test_mask = _extract_masks_from_graph(graph, int(n))
    if train_mask is not None and val_mask is not None and test_mask is not None:
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    data.num_features = int(d)
    data.num_classes = int(nc)
    data = enrich_data_object(
        data,
        dataset_name=dataset_name,
        source_name=f"npz:{npz_path}",
        force_resplit=True,
        split_seed=0,
    )

    return data


def load_public_dataset(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    script_dir = Path(__file__).resolve().parent
    transform = NormalizeFeatures()

    if dataset_name in {"Cora", "CiteSeer", "PubMed"}:
        dataset = Planetoid(
            root=str(script_dir / "tmp" / dataset_name),
            name=dataset_name,
            transform=transform,
        )
        data = dataset[0]
    elif dataset_name == "Flickr":
        dataset = Flickr(
            root=str(script_dir / "tmp" / "Flickr"),
            transform=transform,
        )
        data = dataset[0]
    else:
        raise RuntimeError(f"暂不支持的数据集: {dataset_name}")

    # 先清空官方 public split
    data.train_mask = None
    data.val_mask = None
    data.test_mask = None

    preprocess_fn = pick_first_attr(preprocess, ["preprocess_data"], callable_only=True)
    if preprocess_fn is not None:
        preprocess_kwargs = {
            "data": data.clone(),
            "train_mask_exists": False,
            "target_class": None,
            "train_ratio": 0.7,
            "val_ratio": 0.0,
            "test_ratio": 0.3,
            "seed": 0,
        }
        data = extract_data_from_result(
            preprocess_fn(**filter_kwargs(preprocess_fn, preprocess_kwargs))
        )

    # 关键：无论 preprocess 里做了什么，最终都强制重新按 7:0:3 划分
    data = enrich_data_object(
        data,
        dataset_name=dataset_name,
        source_name="public_dataset",
        force_resplit=True,
        split_seed=0,
    )
    return data





def load_dataset(dataset_name, device):
    dataset_name = normalize_dataset_name(dataset_name)

    npz_path = find_npz_path(dataset_name)
    if npz_path is not None:
        data = load_dataset_from_npz(dataset_name, npz_path)
    else:
        data = load_public_dataset(dataset_name)

    return data.to(device)


def model_forward(model, x, edge_index):
    try:
        return model(x, edge_index)
    except TypeError:
        return model(x)


def choose_classification_loss(logits_or_log_probs, labels):
    if logits_or_log_probs.dim() == 2 and logits_or_log_probs.size(0) > 0:
        sample = logits_or_log_probs[: min(4, logits_or_log_probs.size(0))]
        prob_sum = torch.exp(sample).sum(dim=1)
        if torch.isfinite(prob_sum).all():
            max_error = torch.max(torch.abs(prob_sum - 1.0)).item()
            if max_error < 1e-2:
                return F.nll_loss(logits_or_log_probs, labels)
    return F.cross_entropy(logits_or_log_probs, labels)


def compute_accuracy(output, labels, mask):
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    pred = output.argmax(dim=-1)
    correct = int((pred[mask] == labels[mask]).sum().item())
    return float(correct) / float(total)
def _has_nonempty_mask(data, mask_name):
    mask = getattr(data, mask_name, None)
    if mask is None:
        return False
    return int(mask.sum().item()) > 0


def _mask_accuracy_or_none(output, labels, mask):
    if mask is None:
        return None
    if int(mask.sum().item()) == 0:
        return None
    return compute_accuracy(output, labels, mask)


def _resolve_effective_patch_batch_size(
    requested_batch_size,
    num_train_targets,
    safe_cap=SAFE_MAX_PATCH_TRAIN_BATCH_SIZE,
):
    requested_batch_size = max(1, int(requested_batch_size))
    num_train_targets = int(num_train_targets)
    safe_cap = int(safe_cap)

    effective = max(1, min(requested_batch_size, num_train_targets, safe_cap))
    if effective < requested_batch_size:
        print(
            f"[WARN] patch-train-batch-size={requested_batch_size} 过大，"
            f"共享 Transformer patch 生成器显存开销很高，"
            f"已自动裁剪为 {effective} "
            f"(safe_cap={safe_cap}, num_train_targets={num_train_targets})。"
        )
    return effective


def _is_cuda_memory_or_allocator_error(exc):
    msg = str(exc).lower()
    keywords = [
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "nvml_success == r internal assert failed",
        "cudacachingallocator",
    ]
    return any(k in msg for k in keywords)


def extract_model_from_result(result):
    if isinstance(result, torch.nn.Module):
        return result, {}

    if isinstance(result, dict):
        for key in ["model", "clean_model", "clean_gat_model", "clean_gcn_model", "classifier", "net"]:
            if key in result and isinstance(result[key], torch.nn.Module):
                extra_info = {k: v for k, v in result.items() if not isinstance(v, torch.nn.Module)}
                return result[key], extra_info

    if isinstance(result, (list, tuple)):
        found_model = None
        extra_info = {}
        for item in result:
            if isinstance(item, torch.nn.Module):
                found_model = item
            elif isinstance(item, dict):
                extra_info.update({k: v for k, v in item.items() if not isinstance(v, torch.nn.Module)})
        if found_model is not None:
            return found_model, extra_info

    return None, {}


def fill_clean_metrics(model, data, metrics):
    metrics = dict(metrics)
    model.eval()
    with torch.no_grad():
        out = model_forward(model, data.x, data.edge_index)
        metrics["clean_train_acc"] = _mask_accuracy_or_none(out, data.y, data.train_mask)
        metrics["clean_val_acc"] = _mask_accuracy_or_none(out, data.y, data.val_mask)
        metrics["clean_test_acc"] = _mask_accuracy_or_none(out, data.y, data.test_mask)
    return metrics



def get_default_clean_training_hparams(backbone):
    backbone = normalize_backbone_name(backbone)
    if backbone == "GCN":
        return {
            "hidden_channels": 16,
            "dropout": 0.5,
            "lr": 0.005,
            "weight_decay": 5e-4,
            "heads": 1,
        }
    return {
        "hidden_channels": 8,
        "dropout": 0.6,
        "lr": 0.002,
        "weight_decay": 5e-4,
        "heads": 8,
    }


def build_fallback_model(data, device, backbone="GAT"):
    backbone = normalize_backbone_name(backbone)
    candidate_map = {
        "GAT": ["GAT", "GATNet", "GATModel"],
        "GCN": ["GCN", "GCNNet", "GCNModel"],
    }
    model_cls = pick_first_attr(
        models,
        candidate_map.get(backbone, []) + ["MLP", "MLPNet", "DNN", "DNNModel"],
        callable_only=True,
    )
    if model_cls is None:
        raise RuntimeError(f"models.py 中找不到可用于 {backbone} 的模型类。")

    num_features = int(data.x.size(1))
    num_classes = int(infer_num_classes(data))
    default_hparams = get_default_clean_training_hparams(backbone)

    kwargs = {
        "in_channels": num_features,
        "num_features": num_features,
        "input_dim": num_features,
        "nfeat": num_features,
        "hidden_channels": int(default_hparams["hidden_channels"]),
        "hidden_dim": int(default_hparams["hidden_channels"]),
        "nhid": int(default_hparams["hidden_channels"]),
        "out_channels": num_classes,
        "num_classes": num_classes,
        "output_dim": num_classes,
        "nclass": num_classes,
        "dropout": float(default_hparams["dropout"]),
        "heads": int(default_hparams["heads"]),
        "num_layers": 2,
    }

    model = model_cls(**filter_kwargs(model_cls.__init__, kwargs))
    model = model.to(device)
    return model


def fallback_train_clean_model(data, device, clean_epochs, backbone="GAT"):
    backbone = normalize_backbone_name(backbone)
    default_hparams = get_default_clean_training_hparams(backbone)
    model = build_fallback_model(data, device, backbone=backbone)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(default_hparams["lr"]),
        weight_decay=float(default_hparams["weight_decay"]),
    )

    has_val = _has_nonempty_mask(data, "val_mask")
    best_state = None
    best_val_acc = -1.0
    best_train_loss = float("inf")

    for epoch in range(1, int(clean_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        out = model_forward(model, data.x, data.edge_index)
        loss = choose_classification_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())

        if has_val:
            model.eval()
            with torch.no_grad():
                out = model_forward(model, data.x, data.edge_index)
                val_acc = compute_accuracy(out, data.y, data.val_mask)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        else:
            val_acc = None
            if current_loss <= best_train_loss:
                best_train_loss = current_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if epoch == 1 or epoch % 5 == 0 or epoch == int(clean_epochs):
            if has_val:
                print(
                    f"[Clean {backbone} fallback] epoch {epoch}/{clean_epochs} "
                    f"loss={current_loss:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
                )
            else:
                print(
                    f"[Clean {backbone} fallback] epoch {epoch}/{clean_epochs} "
                    f"loss={current_loss:.4f} (no val set)"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "clean_model_backend": "fallback",
        "clean_model_type": backbone,
        "best_val_acc": float(best_val_acc) if has_val else None,
        "best_train_loss": None if has_val else float(best_train_loss),
    }
    metrics = fill_clean_metrics(model, data, metrics)
    return model, metrics


def train_clean_model(data, dataset_name, device, clean_epochs, backbone="GAT"):
    backbone = normalize_backbone_name(backbone)

    # 如果你把 val 划成 0，则不要再调用依赖 val 的外部 clean trainer
    if not _has_nonempty_mask(data, "val_mask"):
        print(f"[Info] val_mask 为空，跳过外部 {backbone} 干净模型训练脚本，使用内置训练。")
        return fallback_train_clean_model(
            data=data,
            device=device,
            clean_epochs=clean_epochs,
            backbone=backbone,
        )

    train_module = TRAIN_CLEAN_MODULES.get(backbone)
    train_fn_name_candidates = {
        "GAT": [
            "train_clean_gat",
            "train_clean_GAT",
            "train_clean_model",
            "train_model",
            "run_train_clean_gat",
            "main_train_clean_gat",
        ],
        "GCN": [
            "train_clean_GCN",
            "train_clean_gcn",
            "train_clean_model",
            "train_model",
            "run_train_clean_gcn",
            "main_train_clean_gcn",
        ],
    }

    if train_module is not None:
        train_fn = pick_first_attr(
            train_module,
            train_fn_name_candidates.get(backbone, ["train_clean_model", "train_model"]),
            callable_only=True,
        )
        if train_fn is not None:
            train_kwargs = {
                "data": data,
                "dataset_name": dataset_name,
                "dataset": dataset_name,
                "device": device,
                "epochs": int(clean_epochs),
                "output_dir": "./clean_models",
            }
            train_kwargs.update(get_default_clean_training_hparams(backbone))

            try:
                result = train_fn(**filter_kwargs(train_fn, train_kwargs))
                model, extra_info = extract_model_from_result(result)
                if model is not None:
                    model = model.to(device)
                    metrics = fill_clean_metrics(model, data, extra_info)
                    metrics["clean_model_backend"] = "external_module"
                    metrics["clean_model_type"] = backbone
                    return model, metrics
                print(f"[WARN] {backbone} 训练脚本返回值里没有解析出模型，回退到内置训练。")
            except Exception as exc:
                print(f"[WARN] 调用 {backbone} 干净模型训练脚本失败，回退到内置训练。原因: {exc}")
                traceback.print_exc()
        else:
            print(f"[WARN] 无法导入 {backbone} 干净模型训练模块:")

    return fallback_train_clean_model(
        data=data,
        device=device,
        clean_epochs=clean_epochs,
        backbone=backbone,
    )



def _to_python_value(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_value(v) for v in obj]
    return obj


def _flatten_scalar_dict(dct, prefix=""):
    out = {}
    if not isinstance(dct, dict):
        return out

    for key, value in dct.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_scalar_dict(value, prefix=full_key))
        else:
            value = _to_python_value(value)
            if isinstance(value, (list, tuple, dict)):
                out[full_key] = json.dumps(value, ensure_ascii=False)
            else:
                out[full_key] = value
    return out


def _select_nodes_from_mask(data, mask_name, exclude_class=None, max_nodes=None, seed=0):
    mask = getattr(data, mask_name, None)
    if mask is None:
        raise RuntimeError(f"data 中缺少 {mask_name}。")

    node_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    if exclude_class is not None:
        node_indices = node_indices[data.y[node_indices] != int(exclude_class)]

    nodes = [int(x) for x in node_indices.detach().cpu().tolist()]
    if len(nodes) == 0:
        raise RuntimeError(f"{mask_name} 中没有可用节点。exclude_class={exclude_class}")

    if max_nodes is not None and len(nodes) > int(max_nodes):
        rng = random.Random(int(seed))
        rng.shuffle(nodes)
        nodes = sorted(nodes[: int(max_nodes)])

    return nodes


def _make_setting_seed(base_seed, *parts):
    raw = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**31 - 1)


def _sync_trainer_state_from_result(trainer, result):
    if not isinstance(result, dict):
        return

    alias_groups = [
        (
            "best_patch_node_features_logits",
            [
                "best_patch_node_features_logits",
                "patch_node_features_logits",
                "best_patch_node_features",
                "patch_node_features",
                "best_patch_features",
                "patch_features",
            ],
        ),
        (
            "best_patch_adj_logits",
            [
                "best_patch_adj_logits",
                "patch_adj_logits",
                "best_patch_adj",
                "patch_adj",
            ],
        ),
        (
            "best_trigger_feature_logits",
            [
                "best_trigger_feature_logits",
                "trigger_feature_logits",
            ],
        ),
        (
            "best_trigger_adj_logits_rows",
            [
                "best_trigger_adj_logits_rows",
                "trigger_adj_logits_rows",
            ],
        ),
        (
            "best_full_patch_node_features_logits",
            [
                "best_full_patch_node_features_logits",
                "full_patch_node_features_logits",
                "best_full_patch_node_features",
                "full_patch_node_features",
            ],
        ),
        (
            "best_full_patch_adj_logits",
            [
                "best_full_patch_adj_logits",
                "full_patch_adj_logits",
                "best_full_patch_adj",
                "full_patch_adj",
            ],
        ),
    ]

    for target_attr, candidate_keys in alias_groups:
        for key in candidate_keys:
            if key in result and result[key] is not None:
                setattr(trainer, target_attr, result[key])
                break


def _available_training_like_names(obj):
    names = []
    for name in dir(obj):
        low = name.lower()
        if any(k in low for k in ["train", "patch", "trigger", "fit", "optimize"]):
            names.append(name)
    return sorted(set(names))


def _call_candidate_method(obj, module_obj, candidate_names, kwargs, desc):
    method = pick_first_attr(obj, candidate_names, callable_only=True)
    if method is not None:
        print(f"[Trainer] {desc} 使用实例方法: {method.__name__}")
        return method(**filter_kwargs(method, kwargs))

    module_fn = pick_first_attr(module_obj, candidate_names, callable_only=True)
    if module_fn is not None:
        module_kwargs = {"trainer": obj, **kwargs}
        print(f"[Trainer] {desc} 使用模块函数: {module_fn.__name__}")
        return module_fn(**filter_kwargs(module_fn, module_kwargs))

    available = _available_training_like_names(obj)
    raise RuntimeError(
        f"找不到可用于 {desc} 的方法。候选名: {candidate_names}。"
        f" 当前 trainer 可见相关方法: {available}"
    )


def build_patch_trainer(clean_model, data, num_patch_nodes, num_trigger_nodes, device, exp_cfg):
    trainer_cls = pick_first_attr(
        patch_trainer_module,
        [
            "PatchTrainerGNN",
            "PatchTrainer",
            "PatchTrainerGNN_DNN_1_updated",
            "PatchTrainerGNN_DNN_1",
        ],
        callable_only=True,
    )
    if trainer_cls is None:
        raise RuntimeError("在 PatchTrainerGNN_DNN_1_updated.py 中找不到 PatchTrainer 类。")

    trainer_kwargs = {
        "clean_gat_model": clean_model,
        "clean_model": clean_model,
        "num_patch_nodes": int(num_patch_nodes),
        "num_trigger_nodes": int(num_trigger_nodes),
        "num_node_features": int(data.x.size(1)),
        "num_classes": int(infer_num_classes(data)),
        "device": device,
        "target_region_hops": int(exp_cfg.get("target_region_hops", 1)),
        "defense_trials": int(exp_cfg.get("train_defense_trials", exp_cfg.get("defense_trials", DEFAULT_TRAIN_DEFENSE_TRIALS))),
        "defense_drop_prob": float(exp_cfg.get("defense_drop_prob", 0.99)),
        "generator_subgraph_hops": int(exp_cfg.get("generator_subgraph_hops", 1)),
        "shared_generator": True,
        "d_model": int(exp_cfg.get("generator_d_model", 192)),
        "nhead": int(exp_cfg.get("generator_nhead", 8)),
        "num_decoder_layers": int(exp_cfg.get("generator_num_decoder_layers", 4)),
        "dim_feedforward": int(exp_cfg.get("generator_dim_feedforward", 512)),
        "default_train_batch_size": int(exp_cfg.get("train_batch_size", 4)),
        "amp_enabled": True,
        "amp_dtype": "float16",
    }

    trainer = trainer_cls(**filter_kwargs(trainer_cls.__init__, trainer_kwargs))
    trainer.clean_model = clean_model
    trainer.clean_gat_model = clean_model
    trainer.attack_target_class = int(exp_cfg.get("attack_target_class", 0))
    trainer.num_patch_nodes = int(num_patch_nodes)
    trainer.num_trigger_nodes = int(num_trigger_nodes)
    trainer.default_eval_batch_size = int(exp_cfg.get("eval_batch_size", exp_cfg.get("train_batch_size", 4)))
    return trainer

    return trainer


PATCH_TRAIN_METHOD_CANDIDATES = [
    "train_patch_shared",
    "train_shared_patch",
    "train_shared_attack_patch",
    "train_attack_patch",
    "train_patch",
    "optimize_patch",
    "fit_patch",
    "train_patch_generator",
    "run_patch_training",
]


TRIGGER_TRAIN_METHOD_CANDIDATES = [
    "train_trigger_shared",
    "train_shared_trigger",
    "train_trigger",
    "train_universal_trigger",
    "optimize_trigger",
    "fit_trigger",
    "train_trigger_generator",
    "run_trigger_training",
    "train_backdoor_trigger",
]


JOINT_TRAIN_METHOD_CANDIDATES = [
    "train_patch_and_trigger",
    "joint_train",
    "fit",
    "train",
    "run_training",
]


def train_attack_components(
    trainer,
    data,
    train_nodes,
    patch_train_epochs,
    trigger_train_epochs,
    attack_target_class,
    batch_size=None,
):
    requested_batch_size = (
        int(getattr(trainer, "default_train_batch_size", 4))
        if batch_size is None
        else int(batch_size)
    )
    effective_batch_size = _resolve_effective_patch_batch_size(
        requested_batch_size=requested_batch_size,
        num_train_targets=len(train_nodes),
    )

    common_kwargs = {
        "data": data,
        "target_node_indices": train_nodes,
        "target_nodes": train_nodes,
        "train_node_indices": train_nodes,
        "node_indices": train_nodes,
        "attack_target_class": int(attack_target_class),
        "target_class": int(attack_target_class),
        "batch_size": int(effective_batch_size),
        "defense_trials": int(getattr(trainer, "defense_trials", DEFAULT_TRAIN_DEFENSE_TRIALS)),
        "drop_prob": float(getattr(trainer, "defense_drop_prob", 0.99)),
        "num_hops": int(getattr(trainer, "target_region_hops", 1)),
    }

    if int(getattr(trainer, "num_trigger_nodes", 0)) <= 0:
        patch_kwargs = dict(common_kwargs)
        patch_kwargs.update({
            "base_epochs": int(patch_train_epochs),
            "epochs": int(patch_train_epochs),
            "num_epochs": int(patch_train_epochs),
        })
        patch_result = _call_candidate_method(
            trainer,
            patch_trainer_module,
            PATCH_TRAIN_METHOD_CANDIDATES,
            patch_kwargs,
            desc="patch 训练",
        )
        _sync_trainer_state_from_result(trainer, patch_result)
        return {
            "train_mode": "patch_only",
            "patch_train_result": _to_python_value(patch_result),
        }

    patch_method_exists = (
        pick_first_attr(trainer, PATCH_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
        or pick_first_attr(patch_trainer_module, PATCH_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
    )
    trigger_method_exists = (
        pick_first_attr(trainer, TRIGGER_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
        or pick_first_attr(patch_trainer_module, TRIGGER_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
    )
    joint_method_exists = (
        pick_first_attr(trainer, JOINT_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
        or pick_first_attr(patch_trainer_module, JOINT_TRAIN_METHOD_CANDIDATES, callable_only=True) is not None
    )

    separated_exc = None

    if patch_method_exists and trigger_method_exists:
        try:
            patch_kwargs = dict(common_kwargs)
            patch_kwargs.update({
                "base_epochs": int(patch_train_epochs),
                "epochs": int(patch_train_epochs),
                "num_epochs": int(patch_train_epochs),
            })
            patch_result = _call_candidate_method(
                trainer,
                patch_trainer_module,
                PATCH_TRAIN_METHOD_CANDIDATES,
                patch_kwargs,
                desc="patch 训练",
            )
            _sync_trainer_state_from_result(trainer, patch_result)

            trigger_kwargs = dict(common_kwargs)
            trigger_kwargs.update({
                "base_epochs": int(trigger_train_epochs),
                "epochs": int(trigger_train_epochs),
                "num_epochs": int(trigger_train_epochs),
            })
            trigger_result = _call_candidate_method(
                trainer,
                patch_trainer_module,
                TRIGGER_TRAIN_METHOD_CANDIDATES,
                trigger_kwargs,
                desc="trigger 训练",
            )
            _sync_trainer_state_from_result(trainer, trigger_result)

            return {
                "train_mode": "separate_patch_then_trigger",
                "patch_train_result": _to_python_value(patch_result),
                "trigger_train_result": _to_python_value(trigger_result),
            }
        except Exception as exc:
            separated_exc = exc
            print(f"[WARN] 分阶段训练失败。原因: {exc}")
            if _is_cuda_memory_or_allocator_error(exc):
                print(
                    "[WARN] 这是 CUDA/NVML/显存分配错误，不是 trainer 方法不存在。"
                    " 请继续降低 patch batch，例如 16 或 8。"
                )
            traceback.print_exc()

    if joint_method_exists:
        joint_kwargs = dict(common_kwargs)
        joint_kwargs.update({
            "base_epochs": max(int(patch_train_epochs), int(trigger_train_epochs)),
            "epochs": max(int(patch_train_epochs), int(trigger_train_epochs)),
            "num_epochs": max(int(patch_train_epochs), int(trigger_train_epochs)),
            "patch_train_epochs": int(patch_train_epochs),
            "trigger_train_epochs": int(trigger_train_epochs),
        })
        joint_result = _call_candidate_method(
            trainer,
            patch_trainer_module,
            JOINT_TRAIN_METHOD_CANDIDATES,
            joint_kwargs,
            desc="联合训练",
        )
        _sync_trainer_state_from_result(trainer, joint_result)
        return {
            "train_mode": "joint",
            "joint_train_result": _to_python_value(joint_result),
        }

    # 如果分阶段训练已经报了真实错误，就直接抛真实错误，而不是误报“没有方法”
    if separated_exc is not None:
        raise separated_exc

    raise RuntimeError(
        "trainer 中找不到可用的 patch/trigger 分阶段训练方法，也找不到可用的联合训练方法。"
    )




def evaluate_trained_attack(trainer, data, evaluation_node_indices, exp_cfg, patch_mode="per_node", device="cpu"):
    eval_fn = pick_first_attr(
        attack_evaluation_updated,
        ["evaluate_attack", "run_attack_evaluation", "evaluate"],
        callable_only=True,
    )
    if eval_fn is None:
        raise RuntimeError("attack_evaluation_updated.py 中找不到 evaluate_attack。")

    eval_kwargs = {
        "trainer": trainer,
        "data": data,
        "attack_target_class": int(exp_cfg.get("attack_target_class", 0)),
        "evaluation_node_indices": evaluation_node_indices,
        "defense_trials": int(exp_cfg.get("eval_defense_trials", DEFAULT_EVAL_DEFENSE_TRIALS)),
        "drop_prob": float(exp_cfg.get("defense_drop_prob", 0.99)),
        "num_hops": int(exp_cfg.get("target_region_hops", 1)),
        "patch_mode": patch_mode,
        "device": device,
    }
    result = eval_fn(**filter_kwargs(eval_fn, eval_kwargs))
    if isinstance(result, dict):
        return result
    return {"raw_eval_result": _to_python_value(result)}


def apply_runtime_overrides_to_exp_cfg(exp_cfg, args):
    exp_cfg = copy.deepcopy(exp_cfg)

    if args.patch_nodes_list:
        exp_cfg["patch_nodes_list"] = [int(x) for x in args.patch_nodes_list]
    if args.trigger_ratio_levels:
        exp_cfg["trigger_ratio_levels"] = [float(x) for x in args.trigger_ratio_levels]
    if args.patch_train_epochs is not None:
        exp_cfg["patch_train_epochs"] = int(args.patch_train_epochs)
    if args.patch_train_batch_size is not None:
        exp_cfg["train_batch_size"] = int(args.patch_train_batch_size)
    if args.trigger_train_epochs is not None:
        exp_cfg["trigger_train_epochs"] = int(args.trigger_train_epochs)
    if args.train_defense_trials is not None:
        exp_cfg["train_defense_trials"] = int(args.train_defense_trials)
        exp_cfg["defense_trials"] = int(args.train_defense_trials)
    if args.eval_defense_trials is not None:
        exp_cfg["eval_defense_trials"] = int(args.eval_defense_trials)
    if args.attack_target_class is not None:
        exp_cfg["attack_target_class"] = int(args.attack_target_class)
    if args.max_eval_nodes is not None:
        exp_cfg["max_evaluation_nodes"] = int(args.max_eval_nodes)

    return exp_cfg


def save_rows_to_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for key in all_keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    safe_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    safe_row[key] = value
            writer.writerow(safe_row)


def save_json(data_obj, json_path):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_python_value(data_obj), f, ensure_ascii=False, indent=2)
# 断点续跑匹配时需要忽略的字段
# 断点续跑时，仅使用以下 6 个字段判断是否“同一配置”
RESUME_MATCH_FIELDS = [
    "dataset",
    "backbone",
    "num_patch_nodes",
    "num_trigger_nodes",
    "configured_trigger_ratio",
    "actual_trigger_ratio",
    "experiment_phase",
    "phase_run_index",
]

# 兼容不同命名写法
RESUME_FIELD_ALIASES = {
    "dataset": ["dataset"],
    "backbone": ["backbone"],
    "num_patch_nodes": ["num_patch_nodes", "patch_nodes"],
    "num_trigger_nodes": ["num_trigger_nodes", "trigger_nodes"],
    "configured_trigger_ratio": ["configured_trigger_ratio"],
    "actual_trigger_ratio": ["actual_trigger_ratio"],
    "experiment_phase": ["experiment_phase", "phase"],
    "phase_run_index": ["phase_run_index", "repeat_index", "prelude_repeat_index"],
}



def load_existing_rows_from_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception as exc:
        print(f"[WARN] 读取已有 CSV 失败，将按空结果继续。原因: {exc}")
        return []


def _normalize_resume_value(value):
    value = _to_python_value(value)

    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)

    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None

    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    # 尽量把 CSV 里的数字字符串统一转成数值，避免 "0.3000" 和 0.3 匹配不上
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return round(float(text), 10)
        return int(text)
    except Exception:
        return text



def _hash_node_index_list(node_indices):
    raw = ",".join(str(int(x)) for x in node_indices)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
def _try_load_json_dict(raw_text):
    if raw_text is None:
        return None
    raw_text = str(raw_text).strip()
    if not raw_text:
        return None
    try:
        obj = json.loads(raw_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _extract_resume_match_payload(mapping):
    payload = {}
    mapping = mapping if isinstance(mapping, dict) else {}

    for field in RESUME_MATCH_FIELDS:
        value = None
        for alias in RESUME_FIELD_ALIASES.get(field, [field]):
            if alias in mapping:
                value = mapping.get(alias)
                break

        normalized_value = _normalize_resume_value(value)
        if normalized_value is None:
            if field == "experiment_phase":
                normalized_value = "normal"
            elif field == "phase_run_index":
                normalized_value = 0

        payload[field] = normalized_value

    return payload

def build_resume_identity(identity_mapping):
    """
    断点续跑签名只使用以下字段：
    dataset / backbone / num_patch_nodes / num_trigger_nodes /
    configured_trigger_ratio / actual_trigger_ratio
    """
    payload = _extract_resume_match_payload(identity_mapping)

    if any(payload[field] is None for field in RESUME_MATCH_FIELDS):
        return None, None, None

    resume_identity_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    resume_signature = hashlib.md5(
        resume_identity_json.encode("utf-8")
    ).hexdigest()
    return payload, resume_identity_json, resume_signature



def build_experiment_identity(
    dataset_name,
    backbone,
    setting_seed,
    num_patch_nodes,
    num_trigger_nodes,
    configured_trigger_ratio,
    actual_trigger_ratio,
    args,
    exp_cfg,
    num_train_targets,
    num_eval_targets,
    train_nodes_md5,
    eval_nodes_md5,
    experiment_phase="normal",
    phase_run_index=0,
):
    requested_patch_train_batch_size = int(
        args.patch_train_batch_size
        if args.patch_train_batch_size is not None
        else exp_cfg.get("train_batch_size", 4)
    )
    max_eval_nodes = exp_cfg.get("max_evaluation_nodes", None)

    identity = {
        "dataset": str(dataset_name),
        "backbone": str(backbone),
        "experiment_phase": str(experiment_phase),
        "phase_run_index": int(phase_run_index),
        "seed": int(setting_seed),
        "num_patch_nodes": int(num_patch_nodes),
        "num_trigger_nodes": int(num_trigger_nodes),
        "configured_trigger_ratio": float(configured_trigger_ratio),
        "actual_trigger_ratio": float(actual_trigger_ratio),
        "patch_mode": str(args.patch_mode),
        "attack_target_class": int(exp_cfg.get("attack_target_class", 0)),
        "clean_epochs": int(args.clean_epochs),
        "patch_train_epochs": int(exp_cfg.get("patch_train_epochs", 50)),
        "trigger_train_epochs": int(
            exp_cfg.get("trigger_train_epochs", DEFAULT_TRIGGER_TRAIN_EPOCHS)
        ),
        "train_defense_trials": int(
            exp_cfg.get(
                "train_defense_trials",
                exp_cfg.get("defense_trials", DEFAULT_TRAIN_DEFENSE_TRIALS),
            )
        ),
        "eval_defense_trials": int(
            exp_cfg.get("eval_defense_trials", DEFAULT_EVAL_DEFENSE_TRIALS)
        ),
        "defense_drop_prob": float(exp_cfg.get("defense_drop_prob", 0.99)),
        "target_region_hops": int(exp_cfg.get("target_region_hops", 1)),
        "generator_subgraph_hops": int(exp_cfg.get("generator_subgraph_hops", 1)),
        "generator_d_model": int(exp_cfg.get("generator_d_model", 192)),
        "generator_nhead": int(exp_cfg.get("generator_nhead", 8)),
        "generator_num_decoder_layers": int(
            exp_cfg.get("generator_num_decoder_layers", 4)
        ),
        "generator_dim_feedforward": int(
            exp_cfg.get("generator_dim_feedforward", 512)
        ),
        "requested_patch_train_batch_size": int(requested_patch_train_batch_size),
        "max_train_target_nodes": None
        if args.max_train_target_nodes is None
        else int(args.max_train_target_nodes),
        "max_eval_nodes": None if max_eval_nodes is None else int(max_eval_nodes),
        "num_train_targets": int(num_train_targets),
        "num_eval_targets": int(num_eval_targets),
        "train_nodes_md5": str(train_nodes_md5),
        "eval_nodes_md5": str(eval_nodes_md5),
    }

    identity = {k: _normalize_resume_value(v) for k, v in identity.items()}
    identity_json = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    signature = hashlib.md5(identity_json.encode("utf-8")).hexdigest()
    return identity, identity_json, signature


def build_legacy_resume_key(mapping):
    payload = _extract_resume_match_payload(mapping)

    if any(payload[field] is None for field in RESUME_MATCH_FIELDS):
        return None

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )



def build_existing_experiment_index(existing_rows):
    resume_signature_set = set()
    legacy_key_set = set()

    for row in existing_rows:
        existing_identity = (
            _try_load_json_dict(row.get("resume_identity_json"))
            or _try_load_json_dict(row.get("experiment_identity_json"))
            or row
        )

        _, _, recovered_resume_signature = build_resume_identity(existing_identity)
        if recovered_resume_signature is not None:
            resume_signature_set.add(recovered_resume_signature)

        legacy_key = build_legacy_resume_key(existing_identity)
        if legacy_key is not None:
            legacy_key_set.add(legacy_key)

    return resume_signature_set, legacy_key_set



def build_planned_settings(
    dataset_name,
    backbone,
    exp_cfg,
    args,
    train_nodes,
    eval_nodes,
    train_nodes_md5,
    eval_nodes_md5,
    existing_resume_signatures,
    existing_legacy_keys,
    experiment_phase="normal",
    phase_run_index=0,
):
    planned_settings = []

    for num_patch_nodes in exp_cfg["patch_nodes_list"]:
        trigger_candidates = build_num_trigger_nodes_candidates(num_patch_nodes, exp_cfg)

        for trigger_candidate in trigger_candidates:
            num_trigger_nodes = int(trigger_candidate["num_trigger_nodes"])
            configured_trigger_ratio = float(trigger_candidate["configured_trigger_ratio"])
            actual_trigger_ratio = float(trigger_candidate["actual_trigger_ratio"])

            setting_seed = _make_setting_seed(
                args.seed,
                dataset_name,
                backbone,
                num_patch_nodes,
                num_trigger_nodes,
                configured_trigger_ratio,
                experiment_phase,
                phase_run_index,
            )

            identity, identity_json, experiment_signature = build_experiment_identity(
                dataset_name=dataset_name,
                backbone=backbone,
                setting_seed=setting_seed,
                num_patch_nodes=num_patch_nodes,
                num_trigger_nodes=num_trigger_nodes,
                configured_trigger_ratio=configured_trigger_ratio,
                actual_trigger_ratio=actual_trigger_ratio,
                args=args,
                exp_cfg=exp_cfg,
                num_train_targets=len(train_nodes),
                num_eval_targets=len(eval_nodes),
                train_nodes_md5=train_nodes_md5,
                eval_nodes_md5=eval_nodes_md5,
                experiment_phase=experiment_phase,
                phase_run_index=phase_run_index,
            )

            resume_identity, resume_identity_json, resume_signature = build_resume_identity(identity)
            legacy_key = build_legacy_resume_key(identity)
            already_done = (
                resume_signature in existing_resume_signatures
                or (legacy_key is not None and legacy_key in existing_legacy_keys)
            )

            planned_settings.append(
                {
                    "num_patch_nodes": int(num_patch_nodes),
                    "num_trigger_nodes": int(num_trigger_nodes),
                    "configured_trigger_ratio": float(configured_trigger_ratio),
                    "actual_trigger_ratio": float(actual_trigger_ratio),
                    "setting_seed": int(setting_seed),
                    "identity": identity,
                    "identity_json": identity_json,
                    "experiment_signature": experiment_signature,
                    "resume_identity_json": resume_identity_json,
                    "resume_signature": resume_signature,
                    "legacy_key": legacy_key,
                    "already_done": bool(already_done),
                }
            )

    return planned_settings



def execute_planned_settings(
    dataset_name,
    backbone,
    data,
    exp_cfg,
    args,
    train_nodes,
    eval_nodes,
    planned_settings,
    clean_model,
    clean_metrics,
    all_rows,
    existing_resume_signatures,
    existing_legacy_keys,
    results_csv_path,
    results_json_path,
):
    num_existing = sum(1 for item in planned_settings if item["already_done"])
    num_pending = len(planned_settings) - num_existing
    print(
        f"[Resume] {dataset_name}/{backbone}/phase={planned_settings[0]['identity'].get('experiment_phase', 'normal') if planned_settings else 'normal'}: "
        f"total={len(planned_settings)} existing={num_existing} pending={num_pending}"
    )

    if num_pending == 0:
        print(
            f"[Skip] {dataset_name}/{backbone} 当前 phase 的所有配置均已在 {results_csv_path.name} 中存在，跳过当前执行块。"
        )
        return

    for setting in planned_settings:
        if (
            setting["resume_signature"] in existing_resume_signatures
            or (
                setting["legacy_key"] is not None
                and setting["legacy_key"] in existing_legacy_keys
            )
        ):
            print(
                f"[Skip-Exists] dataset={dataset_name} backbone={backbone} "
                f"phase={setting['identity'].get('experiment_phase', 'normal')} "
                f"run_index={setting['identity'].get('phase_run_index', 0)} "
                f"patch_nodes={setting['num_patch_nodes']} "
                f"trigger_nodes={setting['num_trigger_nodes']} "
                f"configured_trigger_ratio={setting['configured_trigger_ratio']:.4f} "
                f"seed={setting['setting_seed']} 已存在，跳过。"
            )
            continue

        num_patch_nodes = int(setting["num_patch_nodes"])
        num_trigger_nodes = int(setting["num_trigger_nodes"])
        configured_trigger_ratio = float(setting["configured_trigger_ratio"])
        actual_trigger_ratio = float(setting["actual_trigger_ratio"])
        setting_seed = int(setting["setting_seed"])

        set_seed(setting_seed)

        print(
            f"\n[Setting] dataset={dataset_name} backbone={backbone} "
            f"phase={setting['identity'].get('experiment_phase', 'normal')} "
            f"run_index={setting['identity'].get('phase_run_index', 0)} "
            f"patch_nodes={num_patch_nodes} trigger_nodes={num_trigger_nodes} "
            f"configured_trigger_ratio={configured_trigger_ratio:.4f} "
            f"actual_trigger_ratio={actual_trigger_ratio:.4f}"
        )

        trainer = build_patch_trainer(
            clean_model=clean_model,
            data=data,
            num_patch_nodes=num_patch_nodes,
            num_trigger_nodes=num_trigger_nodes,
            device=args.device,
            exp_cfg=exp_cfg,
        )

        train_summary = train_attack_components(
            trainer=trainer,
            data=data,
            train_nodes=train_nodes,
            patch_train_epochs=int(exp_cfg.get("patch_train_epochs", 50)),
            trigger_train_epochs=int(
                exp_cfg.get("trigger_train_epochs", DEFAULT_TRIGGER_TRAIN_EPOCHS)
            ),
            attack_target_class=int(exp_cfg.get("attack_target_class", 0)),
            batch_size=args.patch_train_batch_size,
        )

        eval_summary = evaluate_trained_attack(
            trainer=trainer,
            data=data,
            evaluation_node_indices=eval_nodes,
            exp_cfg=exp_cfg,
            patch_mode=args.patch_mode,
            device=args.device,
        )

        row = {
            "dataset": dataset_name,
            "backbone": backbone,
            "seed": int(setting_seed),
            "num_patch_nodes": int(num_patch_nodes),
            "num_trigger_nodes": int(num_trigger_nodes),
            "configured_trigger_ratio": configured_trigger_ratio,
            "actual_trigger_ratio": actual_trigger_ratio,
            "patch_mode": args.patch_mode,
            "attack_target_class": int(exp_cfg.get("attack_target_class", 0)),
            "num_train_targets": int(len(train_nodes)),
            "num_eval_targets": int(len(eval_nodes)),
        }

        row.update(setting["identity"])

        row["experiment_signature"] = setting["experiment_signature"]
        row["experiment_identity_json"] = setting["identity_json"]
        row["resume_signature"] = setting["resume_signature"]
        row["resume_identity_json"] = setting["resume_identity_json"]

        row.update(_flatten_scalar_dict({"clean": clean_metrics}))
        row.update(_flatten_scalar_dict({"train": train_summary}))
        row.update(_flatten_scalar_dict({"eval": eval_summary}))

        all_rows.append(row)
        if setting["resume_signature"] is not None:
            existing_resume_signatures.add(setting["resume_signature"])

        if setting["legacy_key"] is not None:
            existing_legacy_keys.add(setting["legacy_key"])

        save_rows_to_csv(all_rows, results_csv_path)
        save_json(all_rows, results_json_path)

        cleanup_trainer(trainer)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



def run_fixed_gat_prelude(
    args,
    all_rows,
    existing_resume_signatures,
    existing_legacy_keys,
    results_csv_path,
    results_json_path,
):
    print("\n" + "#" * 80)
    print("[Prelude] 开始执行前置 8 轮 GAT 固定参数训练")
    print("#" * 80)

    for phase_run_index in range(1, PRELUDE_REPEAT_COUNT + 1):
        print("\n" + "=" * 80)
        print(f"[Prelude-Round] {phase_run_index}/{PRELUDE_REPEAT_COUNT}")
        print("=" * 80)

        for dataset_name in PRELUDE_DATASETS:
            exp_cfg = apply_runtime_overrides_to_exp_cfg(
                DATASET_ABLATION_CONFIG.get(dataset_name, DATASET_ABLATION_CONFIG["_default"]),
                args,
            )
            exp_cfg["patch_nodes_list"] = [int(PRELUDE_FIXED_PATCH_NODES)]
            exp_cfg["trigger_ratio_levels"] = [float(PRELUDE_FIXED_TRIGGER_RATIO)]
            exp_cfg["eval_defense_trials"] = int(PRELUDE_EVAL_DEFENSE_TRIALS)

            print("\n" + "-" * 80)
            print(
                f"[Prelude-Setting] round={phase_run_index}/{PRELUDE_REPEAT_COUNT} "
                f"dataset={dataset_name} backbone={PRELUDE_BACKBONE} "
                f"patch_nodes={PRELUDE_FIXED_PATCH_NODES} trigger_ratio={PRELUDE_FIXED_TRIGGER_RATIO:.4f} "
                f"eval_k_mc={PRELUDE_EVAL_DEFENSE_TRIALS}"
            )
            print("-" * 80)

            set_seed(args.seed)
            data = load_dataset(dataset_name, device=args.device)
            print(
                f"[Split Check] train={int(data.train_mask.sum())}, "
                f"val={int(data.val_mask.sum())}, "
                f"test={int(data.test_mask.sum())}"
            )

            train_nodes = _select_nodes_from_mask(
                data,
                "train_mask",
                exclude_class=exp_cfg.get("attack_target_class", 0),
                max_nodes=args.max_train_target_nodes,
                seed=args.seed,
            )
            eval_nodes = _select_nodes_from_mask(
                data,
                "test_mask",
                exclude_class=exp_cfg.get("attack_target_class", 0),
                max_nodes=exp_cfg.get("max_evaluation_nodes", None),
                seed=args.seed + phase_run_index,
            )

            print(f"[Prelude-Info] train target nodes: {len(train_nodes)}")
            print(f"[Prelude-Info] full-test eval nodes: {len(eval_nodes)}")

            train_nodes_md5 = _hash_node_index_list(train_nodes)
            eval_nodes_md5 = _hash_node_index_list(eval_nodes)

            planned_settings = build_planned_settings(
                dataset_name=dataset_name,
                backbone=PRELUDE_BACKBONE,
                exp_cfg=exp_cfg,
                args=args,
                train_nodes=train_nodes,
                eval_nodes=eval_nodes,
                train_nodes_md5=train_nodes_md5,
                eval_nodes_md5=eval_nodes_md5,
                existing_resume_signatures=existing_resume_signatures,
                existing_legacy_keys=existing_legacy_keys,
                experiment_phase="prelude_gat",
                phase_run_index=phase_run_index,
            )

            if not planned_settings:
                continue

            set_seed(args.seed)
            clean_model, clean_metrics = train_clean_model(
                data=data,
                dataset_name=dataset_name,
                device=args.device,
                clean_epochs=int(args.clean_epochs),
                backbone=PRELUDE_BACKBONE,
            )

            execute_planned_settings(
                dataset_name=dataset_name,
                backbone=PRELUDE_BACKBONE,
                data=data,
                exp_cfg=exp_cfg,
                args=args,
                train_nodes=train_nodes,
                eval_nodes=eval_nodes,
                planned_settings=planned_settings,
                clean_model=clean_model,
                clean_metrics=clean_metrics,
                all_rows=all_rows,
                existing_resume_signatures=existing_resume_signatures,
                existing_legacy_keys=existing_legacy_keys,
                results_csv_path=results_csv_path,
                results_json_path=results_json_path,
            )

            del clean_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()



def run_ablation_experiments(args):
    datasets = [args.dataset] if args.dataset else (args.datasets if args.datasets else list(DATASET_ORDER))
    datasets = [normalize_dataset_name(x) for x in datasets]

    if args.backbone:
        backbones = [normalize_backbone_name(args.backbone)]
    else:
        backbones = normalize_backbone_names(args.backbones)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = output_dir / "ablation_results.csv"
    results_json_path = output_dir / "ablation_results.json"

    existing_rows = load_existing_rows_from_csv(results_csv_path)
    all_rows = list(existing_rows)
    existing_resume_signatures, existing_legacy_keys = build_existing_experiment_index(existing_rows)

    if existing_rows:
        print(f"[Resume] 已从 {results_csv_path} 读取到 {len(existing_rows)} 条已有记录，重复配置将自动跳过。")
    else:
        print(f"[Resume] 未发现已有结果文件 {results_csv_path}，将从空结果开始。")

    if not getattr(args, "skip_prelude_gat_eight_runs", False):
        run_fixed_gat_prelude(
            args=args,
            all_rows=all_rows,
            existing_resume_signatures=existing_resume_signatures,
            existing_legacy_keys=existing_legacy_keys,
            results_csv_path=results_csv_path,
            results_json_path=results_json_path,
        )

    for dataset_name in datasets:
        exp_cfg = apply_runtime_overrides_to_exp_cfg(
            DATASET_ABLATION_CONFIG.get(dataset_name, DATASET_ABLATION_CONFIG["_default"]),
            args,
        )

        print("\n" + "=" * 80)
        print(f"[Dataset] {dataset_name}")
        print("=" * 80)

        set_seed(args.seed)
        data = load_dataset(dataset_name, device=args.device)
        print(
            f"[Split Check] train={int(data.train_mask.sum())}, "
            f"val={int(data.val_mask.sum())}, "
            f"test={int(data.test_mask.sum())}"
        )

        usable_train_nodes = _select_nodes_from_mask(
            data,
            "train_mask",
            exclude_class=exp_cfg.get("attack_target_class", 0),
            max_nodes=exp_cfg.get("max_evaluation_nodes", None),
            seed=args.seed,
        )
        print(f"[Split Check] usable attack-train nodes={len(usable_train_nodes)}")

        train_nodes = _select_nodes_from_mask(
            data,
            "train_mask",
            exclude_class=exp_cfg.get("attack_target_class", 0),
            max_nodes=args.max_train_target_nodes,
            seed=args.seed,
        )

        eval_nodes = _select_nodes_from_mask(
            data,
            "test_mask",
            exclude_class=exp_cfg.get("attack_target_class", 0),
            max_nodes=exp_cfg.get("max_evaluation_nodes", None),
            seed=args.seed + 1,
        )

        print(f"[Info] train target nodes: {len(train_nodes)}")
        print(f"[Info] eval target nodes : {len(eval_nodes)}")

        train_nodes_md5 = _hash_node_index_list(train_nodes)
        eval_nodes_md5 = _hash_node_index_list(eval_nodes)

        for backbone in backbones:
            print("\n" + "-" * 80)
            print(f"[Backbone] {backbone}")
            print("-" * 80)

            planned_settings = build_planned_settings(
                dataset_name=dataset_name,
                backbone=backbone,
                exp_cfg=exp_cfg,
                args=args,
                train_nodes=train_nodes,
                eval_nodes=eval_nodes,
                train_nodes_md5=train_nodes_md5,
                eval_nodes_md5=eval_nodes_md5,
                existing_resume_signatures=existing_resume_signatures,
                existing_legacy_keys=existing_legacy_keys,
                experiment_phase="normal",
                phase_run_index=0,
            )

            if not planned_settings:
                continue

            set_seed(args.seed)
            clean_model, clean_metrics = train_clean_model(
                data=data,
                dataset_name=dataset_name,
                device=args.device,
                clean_epochs=int(args.clean_epochs),
                backbone=backbone,
            )

            execute_planned_settings(
                dataset_name=dataset_name,
                backbone=backbone,
                data=data,
                exp_cfg=exp_cfg,
                args=args,
                train_nodes=train_nodes,
                eval_nodes=eval_nodes,
                planned_settings=planned_settings,
                clean_model=clean_model,
                clean_metrics=clean_metrics,
                all_rows=all_rows,
                existing_resume_signatures=existing_resume_signatures,
                existing_legacy_keys=existing_legacy_keys,
                results_csv_path=results_csv_path,
                results_json_path=results_json_path,
            )

            del clean_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print("\n所有消融实验已完成。")
    print(f"CSV 结果保存在: {results_csv_path}")
    print(f"JSON 结果保存在: {results_json_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="GAT/GCN 消融实验主程序")

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)

    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--backbones", nargs="*", default=None)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--clean-epochs", type=int, default=200)
    parser.add_argument("--patch-train-epochs", type=int, default=40)
    parser.add_argument("--trigger-train-epochs", type=int, default=40)

    parser.add_argument("--train-defense-trials", type=int, default=8)
    parser.add_argument("--eval-defense-trials", type=int, default=1000)

    parser.add_argument("--patch-nodes-list", nargs="*", type=int, default=None)
    parser.add_argument("--trigger-ratio-levels", nargs="*", type=float, default=None)

    parser.add_argument("--attack-target-class", type=int, default=None)

    parser.add_argument("--max-train-target-nodes", type=int, default=None)
    parser.add_argument("--max-eval-nodes", type=int, default=240)
    parser.add_argument("--patch-train-batch-size", type=int, default=40)

    parser.add_argument("--patch-mode", type=str, choices=["per_node", "fixed"], default="per_node")
    parser.add_argument("--output-dir", type=str, default="./ablation_outputs")
    parser.add_argument("--skip-prelude-gat-eight-runs", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    run_ablation_experiments(args)


if __name__ == "__main__":
    main()

