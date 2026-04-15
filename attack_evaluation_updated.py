# -*- coding: utf-8 -*-
from collections import Counter
import inspect

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, k_hop_subgraph, remove_self_loops, to_undirected


def _filter_kwargs(func, kwargs):
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


def _pick_first_not_none(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _to_python_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _to_python_int(value[0])
    return int(value)


def _infer_device_from_object(obj):
    if obj is None:
        return None
    if isinstance(obj, torch.device):
        return obj
    if isinstance(obj, str):
        return torch.device(obj)
    if torch.is_tensor(obj):
        return obj.device
    if hasattr(obj, "x") and torch.is_tensor(getattr(obj, "x", None)):
        return obj.x.device
    if isinstance(obj, torch.nn.Module):
        try:
            return next(obj.parameters()).device
        except StopIteration:
            return None
    return None


def _normalize_device(device=None, *objects):
    if device is not None:
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    for obj in objects:
        inferred = _infer_device_from_object(obj)
        if inferred is not None:
            return inferred
    return torch.device("cpu")


def _forward_model(model, x, edge_index):
    try:
        return model(x, edge_index)
    except TypeError:
        return model(x)


def _to_undirected_edge_index_only(edge_index, num_nodes):
    result = to_undirected(edge_index, num_nodes=int(num_nodes))
    if isinstance(result, tuple):
        edge_index = result[0]
    else:
        edge_index = result
    return edge_index.long()


def _coalesce_edge_index_only(edge_index, num_nodes):
    result = coalesce(edge_index, None, int(num_nodes), int(num_nodes))
    if isinstance(result, tuple):
        edge_index = result[0]
    else:
        edge_index = result
    return edge_index.long()


def _finalize_binary_undirected_edge_index(edge_index, num_nodes):
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0), dtype=torch.long)

    edge_index = edge_index.long()
    edge_index = _to_undirected_edge_index_only(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = _coalesce_edge_index_only(edge_index, num_nodes)
    return edge_index.long()


def _build_target_region_node_mask(
    original_edge_index,
    target_node_idx,
    num_total_nodes,
    num_original_nodes,
    num_hops=1,
):
    region_nodes, _, _, _ = k_hop_subgraph(
        node_idx=int(target_node_idx),
        num_hops=int(num_hops),
        edge_index=original_edge_index,
        relabel_nodes=False,
        num_nodes=int(num_original_nodes),
    )
    node_mask = torch.zeros(
        int(num_total_nodes),
        dtype=torch.bool,
        device=original_edge_index.device,
    )
    node_mask[region_nodes] = True
    if int(num_total_nodes) > int(num_original_nodes):
        node_mask[int(num_original_nodes): int(num_total_nodes)] = True
    return node_mask


def _split_feature_and_adj_tensors(first, second):
    def _is_square_adj(t):
        return torch.is_tensor(t) and t.dim() == 2 and int(t.size(0)) == int(t.size(1))

    if _is_square_adj(first) and not _is_square_adj(second):
        return second, first
    if _is_square_adj(second) and not _is_square_adj(first):
        return first, second
    return first, second


def _random_drop_edges_in_target_region(
    attacked_edge_index,
    original_edge_index,
    target_node_idx,
    num_total_nodes,
    num_original_nodes,
    drop_prob=0.99,
    num_hops=1,
):
    if attacked_edge_index is None or attacked_edge_index.numel() == 0:
        return attacked_edge_index

    region_node_mask = _build_target_region_node_mask(
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
    keep_pair_mask = torch.rand(unique_pair_ids.size(0), device=attacked_edge_index.device) > float(drop_prob)
    keep_edge_mask = keep_pair_mask[inverse_idx]

    if int(keep_edge_mask.sum().item()) > 0:
        kept_local_edges = local_edges[:, keep_edge_mask]
        new_edge_index = torch.cat([fixed_edges, kept_local_edges], dim=1)
    else:
        new_edge_index = fixed_edges

    return _finalize_binary_undirected_edge_index(new_edge_index, num_total_nodes)


def _is_binary_tensor(tensor, atol=1e-6):
    if tensor is None or (not torch.is_tensor(tensor)):
        return False
    t = tensor.detach()
    if not t.is_floating_point():
        t = t.float()
    in_range = torch.all((t >= -atol) & (t <= 1.0 + atol)).item()
    integer_like = torch.all((t - t.round()).abs() <= atol).item()
    return bool(in_range and integer_like)


def _hard_binarize_patch_outputs(patch_adj_tensor, patch_node_features):
    if _is_binary_tensor(patch_adj_tensor):
        adj_binary = (patch_adj_tensor >= 0.5).float()
    else:
        adj_binary = (torch.sigmoid(patch_adj_tensor) >= 0.5).float()

    if _is_binary_tensor(patch_node_features):
        feature_binary = (patch_node_features >= 0.5).float()
    else:
        feature_binary = (torch.sigmoid(patch_node_features) >= 0.5).float()

    if adj_binary.dim() == 2:
        triu = torch.triu(adj_binary, diagonal=1)
        adj_binary = triu + triu.t()
        adj_binary.fill_diagonal_(0)
    elif adj_binary.dim() == 3:
        triu = torch.triu(adj_binary, diagonal=1)
        adj_binary = triu + triu.transpose(-1, -2)
        diag_idx = torch.arange(adj_binary.size(-1), device=adj_binary.device)
        adj_binary[:, diag_idx, diag_idx] = 0
    else:
        raise ValueError(f"patch_adj_tensor 维度异常: {tuple(adj_binary.shape)}")

    return feature_binary, adj_binary


def _move_patch_tensors_to_data_device(patch_node_features, patch_adj_tensor, data):
    target_device = data.x.device
    if torch.is_tensor(patch_node_features):
        patch_node_features = patch_node_features.to(target_device)
    if torch.is_tensor(patch_adj_tensor):
        patch_adj_tensor = patch_adj_tensor.to(target_device)
    return patch_node_features, patch_adj_tensor


def _build_subgraph_kwargs_for_generator(trainer, data, target_node_idx):
    subgraph_x = None
    subgraph_edge_index = None
    target_local_idx = None

    if hasattr(trainer, "build_target_subgraph"):
        built = trainer.build_target_subgraph(data, int(target_node_idx))
        if isinstance(built, dict):
            subgraph_x = built.get("subgraph_x", built.get("x", None))
            subgraph_edge_index = built.get("subgraph_edge_index", built.get("edge_index", None))
            target_local_idx = built.get("target_local_idx", built.get("mapping", None))
        elif isinstance(built, (list, tuple)) and len(built) >= 3:
            subgraph_x = built[0]
            subgraph_edge_index = built[1]
            target_local_idx = built[2]

    if subgraph_x is None or subgraph_edge_index is None or target_local_idx is None:
        generator_hops = _to_python_int(
            _pick_first_not_none(
                trainer,
                ["generator_subgraph_hops", "generator_num_hops", "target_region_hops", "num_hops"],
                default=1,
            )
        )
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=int(target_node_idx),
            num_hops=max(1, int(generator_hops)),
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=int(data.x.size(0)),
        )
        subgraph_x = data.x[subset]
        subgraph_edge_index = _finalize_binary_undirected_edge_index(sub_edge_index, subgraph_x.size(0))
        target_local_idx = int(mapping.view(-1)[0].item())

    return subgraph_x, subgraph_edge_index, int(target_local_idx)


def _get_attack_patch_tensors_from_trainer(
    trainer,
    data,
    target_node_idx,
    prefer_generator=True,
):
    if prefer_generator:
        if hasattr(trainer, "generate_patch_for_target"):
            first, second = trainer.generate_patch_for_target(
                data=data,
                target_node_idx=int(target_node_idx),
            )
            patch_node_features, patch_adj_tensor = _split_feature_and_adj_tensors(first, second)
            return _move_patch_tensors_to_data_device(patch_node_features, patch_adj_tensor, data)

        patch_generator = _pick_first_not_none(trainer, ["patch_generator"], default=None)
        if patch_generator is not None:
            num_patch_nodes = _to_python_int(
                _pick_first_not_none(trainer, ["num_patch_nodes", "patch_nodes"], default=None)
            )
            if num_patch_nodes is None:
                raise RuntimeError("trainer 中缺少 num_patch_nodes。")

            subgraph_x, subgraph_edge_index, target_local_idx = _build_subgraph_kwargs_for_generator(
                trainer=trainer,
                data=data,
                target_node_idx=int(target_node_idx),
            )

            call_target = patch_generator.forward if hasattr(patch_generator, "forward") else patch_generator
            kwargs = {
                "subgraph_x": subgraph_x,
                "subgraph_edge_index": subgraph_edge_index,
                "target_local_idx": int(target_local_idx),
                "num_patch_nodes": int(num_patch_nodes),
                "x": subgraph_x,
                "edge_index": subgraph_edge_index,
                "target_node_idx": int(target_local_idx),
                "full_graph_x": data.x,
                "full_graph_edge_index": data.edge_index,
            }
            kwargs = _filter_kwargs(call_target, kwargs)
            first, second = patch_generator(**kwargs)
            patch_node_features, patch_adj_tensor = _split_feature_and_adj_tensors(first, second)
            return _move_patch_tensors_to_data_device(patch_node_features, patch_adj_tensor, data)

    patch_node_features = _pick_first_not_none(
        trainer,
        [
            "best_patch_node_features_logits",
            "patch_node_features_logits",
            "best_patch_node_features",
            "best_patch_features",
            "patch_node_features",
            "learned_patch_node_features",
            "optimized_patch_node_features",
        ],
        default=None,
    )
    patch_adj_tensor = _pick_first_not_none(
        trainer,
        [
            "best_patch_adj_logits",
            "patch_adj_logits",
            "best_patch_adj",
            "patch_adj",
            "patch_adj_tensor",
            "learned_patch_adj",
            "optimized_patch_adj",
        ],
        default=None,
    )

    if patch_node_features is None or patch_adj_tensor is None:
        raise RuntimeError("trainer 中找不到攻击 patch，也无法通过共享生成器实时生成。")

    return _move_patch_tensors_to_data_device(patch_node_features, patch_adj_tensor, data)


def _get_full_patch_tensors_from_trainer(
    trainer,
    data,
    target_node_idx,
    attack_patch_tensors=None,
    prefer_generator=True,
):
    if prefer_generator and hasattr(trainer, "generate_full_patch_for_target"):
        first, second = trainer.generate_full_patch_for_target(
            data=data,
            target_node_idx=int(target_node_idx),
            binary=True,
        )
        full_patch_node_features, full_patch_adj_tensor = _split_feature_and_adj_tensors(first, second)
        return _move_patch_tensors_to_data_device(full_patch_node_features, full_patch_adj_tensor, data)

    trigger_feature_logits = _pick_first_not_none(
        trainer,
        ["best_trigger_feature_logits", "trigger_feature_logits"],
        default=None,
    )
    trigger_adj_logits_rows = _pick_first_not_none(
        trainer,
        ["best_trigger_adj_logits_rows", "trigger_adj_logits_rows"],
        default=None,
    )

    if (
        prefer_generator
        and trigger_feature_logits is not None
        and trigger_adj_logits_rows is not None
        and hasattr(trainer, "build_full_patch_logits")
    ):
        if attack_patch_tensors is None:
            attack_patch_tensors = _get_attack_patch_tensors_from_trainer(
                trainer=trainer,
                data=data,
                target_node_idx=int(target_node_idx),
                prefer_generator=True,
            )

        attack_patch_node_features, attack_patch_adj = attack_patch_tensors

        if hasattr(trainer, "export_fixed_attack_patch_binary"):
            fixed_attack_adj_binary, fixed_attack_features_binary = trainer.export_fixed_attack_patch_binary(
                attack_patch_adj,
                attack_patch_node_features,
            )
        else:
            fixed_attack_features_binary, fixed_attack_adj_binary = _hard_binarize_patch_outputs(
                attack_patch_adj,
                attack_patch_node_features,
            )

        if hasattr(trainer, "export_full_patch_binary"):
            first, second = trainer.export_full_patch_binary(
                fixed_attack_adj_binary=fixed_attack_adj_binary,
                fixed_attack_features_binary=fixed_attack_features_binary,
                trigger_feature_logits=trigger_feature_logits,
                trigger_adj_logits_rows=trigger_adj_logits_rows,
            )
            full_feature_binary, full_adj_binary = _split_feature_and_adj_tensors(first, second)
        else:
            full_adj_logits, full_feature_logits = trainer.build_full_patch_logits(
                fixed_attack_adj_binary=fixed_attack_adj_binary,
                fixed_attack_features_binary=fixed_attack_features_binary,
                trigger_feature_logits=trigger_feature_logits,
                trigger_adj_logits_rows=trigger_adj_logits_rows,
            )
            full_feature_binary, full_adj_binary = _hard_binarize_patch_outputs(
                full_adj_logits,
                full_feature_logits,
            )

        return _move_patch_tensors_to_data_device(full_feature_binary, full_adj_binary, data)

    full_patch_node_features = _pick_first_not_none(
        trainer,
        [
            "best_full_patch_node_features_logits",
            "best_full_patch_node_features",
            "full_patch_node_features_logits",
            "full_patch_node_features",
        ],
        default=None,
    )
    full_patch_adj_tensor = _pick_first_not_none(
        trainer,
        [
            "best_full_patch_adj_logits",
            "best_full_patch_adj",
            "full_patch_adj_logits",
            "full_patch_adj",
        ],
        default=None,
    )

    if full_patch_node_features is None or full_patch_adj_tensor is None:
        raise RuntimeError("trainer 中找不到 full patch，也无法基于共享生成器和共享 trigger 动态构造。")

    return _move_patch_tensors_to_data_device(full_patch_node_features, full_patch_adj_tensor, data)


def collect_random_defense_predictions(
    clean_model,
    features_with_patch,
    edge_index_with_patch,
    original_edge_index,
    target_node_idx,
    num_original_nodes,
    defense_trials=100,
    drop_prob=0.99,
    num_hops=1,
):
    pred_list = []
    clean_model.eval()

    with torch.no_grad():
        for _ in range(int(defense_trials)):
            defended_edge_index = _random_drop_edges_in_target_region(
                attacked_edge_index=edge_index_with_patch,
                original_edge_index=original_edge_index,
                target_node_idx=target_node_idx,
                num_total_nodes=features_with_patch.size(0),
                num_original_nodes=num_original_nodes,
                drop_prob=drop_prob,
                num_hops=num_hops,
            )
            out = _forward_model(clean_model, features_with_patch, defended_edge_index)
            pred = int(out[int(target_node_idx)].argmax(dim=-1).item())
            pred_list.append(pred)

    pred_counter = Counter(pred_list)
    return pred_list, pred_counter


def _resolve_evaluation_nodes(data, evaluation_node_indices=None, attack_target_class=None):
    if evaluation_node_indices is not None:
        if torch.is_tensor(evaluation_node_indices):
            node_indices = [int(x) for x in evaluation_node_indices.view(-1).detach().cpu().tolist()]
        else:
            node_indices = [int(x) for x in list(evaluation_node_indices)]
        if len(node_indices) == 0:
            raise RuntimeError("evaluation_node_indices 为空，无法执行统一评估。")
        return node_indices

    if not hasattr(data, "test_mask") or data.test_mask is None:
        raise RuntimeError("data 中缺少 test_mask，无法按 test 集统一评估。")

    test_nodes = torch.nonzero(data.test_mask, as_tuple=False).view(-1)
    if attack_target_class is not None:
        test_nodes = test_nodes[data.y[test_nodes] != int(attack_target_class)]

    node_indices = [int(x) for x in test_nodes.detach().cpu().tolist()]
    if len(node_indices) == 0:
        raise RuntimeError("test 集中没有可用于攻击评估的节点。")
    return node_indices


def _apply_patch_only(
    trainer,
    data,
    target_node_idx,
    patch_adj_tensor,
    patch_node_features,
    clean_edge_index_override=None,
    structure_keep_mask=None,
):
    if hasattr(trainer, "_apply_patch_to_graph"):
        apply_kwargs = {
            "original_features": data.x,
            "original_edge_index": data.edge_index,
            "target_node_idx": int(target_node_idx),
            "patch_adj_tensor_for_structure": patch_adj_tensor,
            "patch_node_features": patch_node_features,
            "clean_edge_index_override": clean_edge_index_override,
            "structure_keep_mask": structure_keep_mask,
        }
        apply_kwargs = _filter_kwargs(trainer._apply_patch_to_graph, apply_kwargs)
        attacked_x, attacked_edge_index = trainer._apply_patch_to_graph(**apply_kwargs)
    else:
        raise RuntimeError("trainer 中缺少 _apply_patch_to_graph，无法评估攻击 patch。")

    attacked_edge_index = _finalize_binary_undirected_edge_index(
        attacked_edge_index,
        attacked_x.size(0),
    )
    return attacked_x, attacked_edge_index


def _apply_full_patch(
    trainer,
    data,
    target_node_idx,
    full_patch_adj_tensor,
    full_patch_node_features,
    clean_edge_index_override=None,
    structure_keep_mask=None,
):
    current_features = data.x
    current_edge_index = data.edge_index

    if not hasattr(trainer, "_apply_patch_to_graph"):
        raise RuntimeError("trainer 中缺少 _apply_patch_to_graph。")

    patch_kwargs = {
        "original_features": current_features,
        "original_edge_index": current_edge_index,
        "target_node_idx": int(target_node_idx),
        "patch_adj_tensor_for_structure": full_patch_adj_tensor,
        "patch_node_features": full_patch_node_features,
        "clean_edge_index_override": clean_edge_index_override,
        "structure_keep_mask": structure_keep_mask,
    }
    patch_kwargs = _filter_kwargs(trainer._apply_patch_to_graph, patch_kwargs)
    current_features, current_edge_index = trainer._apply_patch_to_graph(**patch_kwargs)

    if int(getattr(trainer, "num_trigger_nodes", 0)) > 0:
        if not hasattr(trainer, "_apply_trigger_to_graph"):
            raise RuntimeError("trainer 中缺少 _apply_trigger_to_graph。")
        trigger_kwargs = {
            "original_features": current_features,
            "original_edge_index": current_edge_index,
            "target_node_idx": int(target_node_idx),
            "patch_adj_tensor_for_structure": full_patch_adj_tensor,
            "patch_node_features": full_patch_node_features,
            "structure_keep_mask": structure_keep_mask,
        }
        trigger_kwargs = _filter_kwargs(trainer._apply_trigger_to_graph, trigger_kwargs)
        current_features, current_edge_index = trainer._apply_trigger_to_graph(**trigger_kwargs)

    current_edge_index = _finalize_binary_undirected_edge_index(
        current_edge_index,
        current_features.size(0),
    )
    return current_features, current_edge_index


def _safe_rate(numerator, denominator):
    return float(numerator) / float(max(1, denominator))


def _prepare_unique_undirected_edges(edge_index):
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))
    row, col = edge_index
    mask = row < col
    return torch.stack([row[mask], col[mask]], dim=0)


def _sample_global_random_defense_edge_indices(
    original_edge_index,
    num_original_nodes,
    k_mc,
    drop_prob,
    device,
):
    with torch.no_grad():
        base_edge_index = _finalize_binary_undirected_edge_index(
            original_edge_index.to(device),
            int(num_original_nodes),
        )
        unique_edges = _prepare_unique_undirected_edges(base_edge_index)

        if unique_edges.numel() == 0:
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            return [empty_edge_index for _ in range(int(k_mc))]

        num_unique_edges = int(unique_edges.size(1))
        keep_mask = torch.rand(int(k_mc), num_unique_edges, device=device) > float(drop_prob)

        defended_edge_indices = []
        for mc_idx in range(int(k_mc)):
            kept_edges = unique_edges[:, keep_mask[mc_idx]]
            if kept_edges.numel() > 0:
                directed_kept_edges = torch.cat([kept_edges, kept_edges.flip(0)], dim=1)
            else:
                directed_kept_edges = torch.empty((2, 0), dtype=torch.long, device=device)

            defended_edge_indices.append(
                _finalize_binary_undirected_edge_index(
                    directed_kept_edges,
                    int(num_original_nodes),
                )
            )
        return defended_edge_indices


def _sample_patch_structure_keep_masks(
    num_patch_nodes,
    k_mc,
    drop_prob,
    device,
):
    with torch.no_grad():
        patch_size_with_target = int(num_patch_nodes) + 1
        tri_idx = torch.triu_indices(
            patch_size_with_target,
            patch_size_with_target,
            offset=1,
            device=device,
        )
        num_pairs = int(tri_idx.size(1))

        keep_mask = torch.rand(int(k_mc), num_pairs, device=device) > float(drop_prob)

        structure_masks = torch.zeros(
            int(k_mc),
            patch_size_with_target,
            patch_size_with_target,
            dtype=torch.bool,
            device=device,
        )
        structure_masks[:, tri_idx[0], tri_idx[1]] = keep_mask
        structure_masks[:, tri_idx[1], tri_idx[0]] = keep_mask
        diag = torch.arange(patch_size_with_target, device=device)
        structure_masks[:, diag, diag] = False
        return structure_masks


def _prepare_evaluation_defense_cache(
    trainer,
    data,
    defense_trials,
    drop_prob,
    device,
):
    clean_edge_indices = _sample_global_random_defense_edge_indices(
        original_edge_index=data.edge_index,
        num_original_nodes=int(data.x.size(0)),
        k_mc=int(defense_trials),
        drop_prob=float(drop_prob),
        device=device,
    )

    num_patch_nodes = int(_pick_first_not_none(trainer, ["num_patch_nodes", "patch_nodes"], default=0) or 0)
    if num_patch_nodes > 0:
        patch_structure_masks = _sample_patch_structure_keep_masks(
            num_patch_nodes=num_patch_nodes,
            k_mc=int(defense_trials),
            drop_prob=float(drop_prob),
            device=device,
        )
    else:
        patch_structure_masks = None

    return {
        "num_mc": int(defense_trials),
        "clean_edge_indices": clean_edge_indices,
        "patch_structure_masks": patch_structure_masks,
    }


def _infer_num_classes(data, trainer=None):
    if hasattr(data, "num_classes") and data.num_classes is not None:
        try:
            return int(data.num_classes)
        except Exception:
            pass
    if trainer is not None:
        val = _pick_first_not_none(trainer, ["num_classes"], default=None)
        if val is not None:
            return int(val)
    if hasattr(data, "y") and data.y is not None:
        return int(data.y.max().item()) + 1
    raise RuntimeError("无法推断类别数。")


def _generate_attack_patch_batch_for_nodes(trainer, data, target_nodes):
    target_nodes = [int(x) for x in target_nodes]

    # 优先走 trainer 的第一阶段冻结 patch 批次生成接口
    if hasattr(trainer, "generate_frozen_patch_batch_for_targets"):
        with torch.no_grad():
            patch_feature_batch, patch_adj_batch = trainer.generate_frozen_patch_batch_for_targets(
                data=data,
                target_node_indices=target_nodes,
            )
        return _move_patch_tensors_to_data_device(patch_feature_batch, patch_adj_batch, data)

    # 次优先：走 trainer 当前 patch 生成接口
    if hasattr(trainer, "generate_patch_batch_for_targets"):
        with torch.no_grad():
            patch_feature_batch, patch_adj_batch = trainer.generate_patch_batch_for_targets(
                data=data,
                target_node_indices=target_nodes,
            )
        return _move_patch_tensors_to_data_device(patch_feature_batch, patch_adj_batch, data)


    # 回退到逐点生成
    feature_list = []
    adj_list = []
    with torch.no_grad():
        for node_idx in target_nodes:
            patch_node_features, patch_adj_tensor = _get_attack_patch_tensors_from_trainer(
                trainer=trainer,
                data=data,
                target_node_idx=int(node_idx),
                prefer_generator=True,
            )
            feature_list.append(patch_node_features)
            adj_list.append(patch_adj_tensor)

    return torch.stack(feature_list, dim=0), torch.stack(adj_list, dim=0)


def _generate_full_patch_batch_for_nodes(
    trainer,
    data,
    target_nodes,
    attack_patch_feature_batch=None,
    attack_patch_adj_batch=None,
):
    target_nodes = [int(x) for x in target_nodes]

    trigger_feature_logits = _pick_first_not_none(
        trainer,
        ["best_trigger_feature_logits", "trigger_feature_logits"],
        default=None,
    )
    trigger_adj_logits_rows = _pick_first_not_none(
        trainer,
        ["best_trigger_adj_logits_rows", "trigger_adj_logits_rows"],
        default=None,
    )

    # 最优路径：
    # 已经拿到了同一个 chunk 的 attack patch batch，
    # 那就直接批次构造 full patch，避免再次逐点生成
    if (
        attack_patch_feature_batch is not None
        and attack_patch_adj_batch is not None
        and trigger_feature_logits is not None
        and trigger_adj_logits_rows is not None
        and hasattr(trainer, "build_full_patch_logits_batch")
    ):
        with torch.no_grad():
            if hasattr(trainer, "export_fixed_attack_patch_binary"):
                fixed_attack_adj_binary_batch, fixed_attack_features_binary_batch = trainer.export_fixed_attack_patch_binary(
                    attack_patch_adj_batch,
                    attack_patch_feature_batch,
                )
            else:
                fixed_attack_features_binary_batch, fixed_attack_adj_binary_batch = _hard_binarize_patch_outputs(
                    attack_patch_adj_batch,
                    attack_patch_feature_batch,
                )

            if hasattr(trainer, "export_full_patch_binary_batch"):
                full_adj_binary_batch, full_feature_binary_batch = trainer.export_full_patch_binary_batch(
                    fixed_attack_adj_binary_batch=fixed_attack_adj_binary_batch,
                    fixed_attack_features_binary_batch=fixed_attack_features_binary_batch,
                    trigger_feature_logits=trigger_feature_logits,
                    trigger_adj_logits_rows=trigger_adj_logits_rows,
                )
                return _move_patch_tensors_to_data_device(
                    full_feature_binary_batch,
                    full_adj_binary_batch,
                    data,
                )

            full_adj_logits_batch, full_feature_logits_batch = trainer.build_full_patch_logits_batch(
                fixed_attack_adj_binary_batch=fixed_attack_adj_binary_batch,
                fixed_attack_features_binary_batch=fixed_attack_features_binary_batch,
                trigger_feature_logits=trigger_feature_logits,
                trigger_adj_logits_rows=trigger_adj_logits_rows,
            )
            full_feature_binary_batch, full_adj_binary_batch = _hard_binarize_patch_outputs(
                full_adj_logits_batch,
                full_feature_logits_batch,
            )
            return _move_patch_tensors_to_data_device(
                full_feature_binary_batch,
                full_adj_binary_batch,
                data,
            )

    # 次优路径：trainer 直接支持 full patch 的 batch 生成
    if hasattr(trainer, "generate_full_patch_batch_for_targets"):
        with torch.no_grad():
            full_feature_batch, full_adj_batch = trainer.generate_full_patch_batch_for_targets(
                data=data,
                target_node_indices=target_nodes,
                binary=True,
            )
        return _move_patch_tensors_to_data_device(full_feature_batch, full_adj_batch, data)

    # 回退：逐点生成
    full_feature_list = []
    full_adj_list = []
    with torch.no_grad():
        for local_idx, node_idx in enumerate(target_nodes):
            attack_patch_tensors = None
            if attack_patch_feature_batch is not None and attack_patch_adj_batch is not None:
                attack_patch_tensors = (
                    attack_patch_feature_batch[local_idx],
                    attack_patch_adj_batch[local_idx],
                )

            full_patch_node_features, full_patch_adj_tensor = _get_full_patch_tensors_from_trainer(
                trainer=trainer,
                data=data,
                target_node_idx=int(node_idx),
                attack_patch_tensors=attack_patch_tensors,
                prefer_generator=True,
            )
            full_feature_list.append(full_patch_node_features)
            full_adj_list.append(full_patch_adj_tensor)

    return torch.stack(full_feature_list, dim=0), torch.stack(full_adj_list, dim=0)



def _accumulate_prediction_counts_from_batched_graphs(
    trainer,
    clean_model,
    data,
    target_nodes,
    patch_adj_batch,
    patch_feature_batch,
    clean_edge_indices,
    patch_structure_masks,
    num_classes,
    use_full_patch=False,
):
    chunk_size = int(len(target_nodes))
    pred_counts = torch.zeros(chunk_size, int(num_classes), dtype=torch.int32)

    has_batch_helpers = (
        hasattr(trainer, "_build_patch_only_data_list")
        and hasattr(trainer, "_build_full_patch_data_list")
        and hasattr(trainer, "_batch_graph_list_and_collect_target_indices")
    )

    with torch.no_grad():
        for mc_idx in range(int(len(clean_edge_indices))):
            clean_edge_index_mc = clean_edge_indices[mc_idx]
            structure_keep_mask_mc = None if patch_structure_masks is None else patch_structure_masks[mc_idx]

            if has_batch_helpers:
                if use_full_patch:
                    graph_list = trainer._build_full_patch_data_list(
                        data=data,
                        target_nodes=target_nodes,
                        full_patch_adj_batch=patch_adj_batch,
                        full_patch_feature_batch=patch_feature_batch,
                        clean_edge_index=clean_edge_index_mc,
                        structure_keep_mask=structure_keep_mask_mc,
                    )
                else:
                    graph_list = trainer._build_patch_only_data_list(
                        data=data,
                        target_nodes=target_nodes,
                        patch_adj_batch=patch_adj_batch,
                        patch_feature_batch=patch_feature_batch,
                        clean_edge_index=clean_edge_index_mc,
                        structure_keep_mask=structure_keep_mask_mc,
                    )

                batched_graph, global_target_indices = trainer._batch_graph_list_and_collect_target_indices(graph_list)
                out = _forward_model(clean_model, batched_graph.x, batched_graph.edge_index)
                preds = out[global_target_indices].argmax(dim=1).detach().cpu()
                pred_counts += F.one_hot(preds.long(), num_classes=int(num_classes)).to(torch.int32)

                del graph_list, batched_graph, global_target_indices, out, preds
            else:
                preds_list = []
                for i, node_idx in enumerate(target_nodes):
                    if use_full_patch:
                        attacked_x, attacked_edge_index = _apply_full_patch(
                            trainer=trainer,
                            data=data,
                            target_node_idx=int(node_idx),
                            full_patch_adj_tensor=patch_adj_batch[i],
                            full_patch_node_features=patch_feature_batch[i],
                            clean_edge_index_override=clean_edge_index_mc,
                            structure_keep_mask=structure_keep_mask_mc,
                        )
                    else:
                        attacked_x, attacked_edge_index = _apply_patch_only(
                            trainer=trainer,
                            data=data,
                            target_node_idx=int(node_idx),
                            patch_adj_tensor=patch_adj_batch[i],
                            patch_node_features=patch_feature_batch[i],
                            clean_edge_index_override=clean_edge_index_mc,
                            structure_keep_mask=structure_keep_mask_mc,
                        )
                    out = _forward_model(clean_model, attacked_x, attacked_edge_index)
                    preds_list.append(int(out[int(node_idx)].argmax(dim=-1).item()))
                    del attacked_x, attacked_edge_index, out

                preds = torch.tensor(preds_list, dtype=torch.long)
                pred_counts += F.one_hot(preds.long(), num_classes=int(num_classes)).to(torch.int32)
                del preds

    return pred_counts


def evaluate_attack(
    trainer,
    data,
    target_node_idx=None,
    attack_target_class=None,
    evaluation_node_indices=None,
    defense_trials=10000,
    drop_prob=0.99,
    num_hops=1,
    patch_mode="per_node",
    device=None,
):
    eval_device = _normalize_device(device, data, trainer)
    if hasattr(data, "to"):
        data = data.to(eval_device)

    if attack_target_class is None:
        attack_target_class = _to_python_int(
            _pick_first_not_none(
                trainer,
                ["attack_target_class", "target_class", "attack_class"],
                default=None,
            )
        )

    if defense_trials is None:
        defense_trials = _to_python_int(
            _pick_first_not_none(trainer, ["defense_trials"], default=10000)
        )
    if drop_prob is None:
        drop_prob = float(
            _pick_first_not_none(
                trainer,
                ["defense_drop_prob", "drop_prob"],
                default=0.98,
            )
        )
    if num_hops is None:
        num_hops = _to_python_int(
            _pick_first_not_none(
                trainer,
                ["target_region_hops", "num_hops"],
                default=1,
            )
        )

    defense_trials = max(1, int(defense_trials))
    drop_prob = float(drop_prob)
    num_hops = max(1, int(num_hops))

    clean_model = _pick_first_not_none(
        trainer,
        ["clean_gat_model", "clean_model", "model"],
        default=None,
    )
    if clean_model is None:
        raise RuntimeError("trainer 中找不到 clean model。")
    clean_model = clean_model.to(eval_device).eval()

    if evaluation_node_indices is None and target_node_idx is not None:
        evaluation_node_indices = [int(target_node_idx)]

    evaluation_nodes = _resolve_evaluation_nodes(
        data=data,
        evaluation_node_indices=evaluation_node_indices,
        attack_target_class=attack_target_class,
    )

    trigger_params_ready = (
        _pick_first_not_none(trainer, ["best_trigger_feature_logits", "trigger_feature_logits"], default=None)
        is not None
        and _pick_first_not_none(trainer, ["best_trigger_adj_logits_rows", "trigger_adj_logits_rows"], default=None)
        is not None
    )
    stored_full_patch_ready = (
        _pick_first_not_none(
            trainer,
            ["best_full_patch_node_features_logits", "best_full_patch_node_features"],
            default=None,
        )
        is not None
        and _pick_first_not_none(
            trainer,
            ["best_full_patch_adj_logits", "best_full_patch_adj"],
            default=None,
        )
        is not None
    )
    trigger_eval_enabled = bool(
        int(getattr(trainer, "num_trigger_nodes", 0)) > 0
        and (trigger_params_ready or stored_full_patch_ready or hasattr(trainer, "generate_full_patch_for_target"))
    )

    num_original_nodes = int(data.x.size(0))
    num_eval_nodes = int(len(evaluation_nodes))
    num_classes = _infer_num_classes(data, trainer=trainer)
    eval_batch_size = int(
        _pick_first_not_none(
            trainer,
            ["default_eval_batch_size", "default_train_batch_size"],
            default=4,
        )
    )
    eval_batch_size = max(1, min(eval_batch_size, num_eval_nodes))

    print(
        f"[Eval] pre-sampling global defense cache: "
        f"mc={defense_trials} drop_prob={drop_prob:.4f} eval_nodes={num_eval_nodes}"
    )
    defense_cache = _prepare_evaluation_defense_cache(
        trainer=trainer,
        data=data,
        defense_trials=defense_trials,
        drop_prob=drop_prob,
        device=eval_device,
    )
    clean_edge_indices = defense_cache["clean_edge_indices"]
    patch_structure_masks = defense_cache["patch_structure_masks"]

    evaluation_nodes_tensor = torch.tensor(
        evaluation_nodes,
        dtype=torch.long,
        device=eval_device,
    )
    true_labels_tensor = data.y[evaluation_nodes_tensor]
    true_labels_cpu = true_labels_tensor.detach().cpu().long()

    clean_pred_counts = torch.zeros(num_eval_nodes, num_classes, dtype=torch.int32)
    attack_pred_counts = torch.zeros(num_eval_nodes, num_classes, dtype=torch.int32)
    trigger_pred_counts = torch.zeros(num_eval_nodes, num_classes, dtype=torch.int32) if trigger_eval_enabled else None

    progress_interval_mc = max(1, defense_trials // 5)

    # 1) clean graph：同一份随机删边后的全图，对所有评估节点统一统计
    with torch.no_grad():
        for mc_idx in range(defense_trials):
            out = _forward_model(clean_model, data.x, clean_edge_indices[mc_idx])
            preds = out[evaluation_nodes_tensor].argmax(dim=1).detach().cpu()
            clean_pred_counts += F.one_hot(preds.long(), num_classes=num_classes).to(torch.int32)

            if (
                mc_idx == 0
                or (mc_idx + 1) % progress_interval_mc == 0
                or (mc_idx + 1) == defense_trials
            ):
                print(f"[Eval-CleanCache] mc {mc_idx + 1}/{defense_trials}")

            del out, preds

    cached_attack_patch = None
    cached_full_patch = None
    if patch_mode == "fixed":
        patch_seed_node = int(evaluation_nodes[0])
        with torch.no_grad():
            cached_attack_patch = _get_attack_patch_tensors_from_trainer(
                trainer=trainer,
                data=data,
                target_node_idx=patch_seed_node,
                prefer_generator=True,
            )
            if trigger_eval_enabled:
                cached_full_patch = _get_full_patch_tensors_from_trainer(
                    trainer=trainer,
                    data=data,
                    target_node_idx=patch_seed_node,
                    attack_patch_tensors=cached_attack_patch,
                    prefer_generator=True,
                )

    num_chunks = (num_eval_nodes + eval_batch_size - 1) // eval_batch_size

    # 2) attack patch / full patch：缓存好的 mc 图结构 + patch mask，逐块评估所有节点
    for chunk_idx, start in enumerate(range(0, num_eval_nodes, eval_batch_size), 1):
        end = min(start + eval_batch_size, num_eval_nodes)
        chunk_nodes = evaluation_nodes[start:end]
        chunk_size = end - start

        print(
            f"[Eval-Attack] chunk {chunk_idx}/{num_chunks} "
            f"nodes={start + 1}-{end}/{num_eval_nodes}"
        )

        with torch.no_grad():
            if patch_mode == "fixed":
                attack_patch_node_features, attack_patch_adj_tensor = cached_attack_patch
                attack_patch_feature_batch = attack_patch_node_features.unsqueeze(0).expand(chunk_size, -1, -1)
                attack_patch_adj_batch = attack_patch_adj_tensor.unsqueeze(0).expand(chunk_size, -1, -1)

                if trigger_eval_enabled:
                    full_patch_node_features, full_patch_adj_tensor = cached_full_patch
                    full_patch_feature_batch = full_patch_node_features.unsqueeze(0).expand(chunk_size, -1, -1)
                    full_patch_adj_batch = full_patch_adj_tensor.unsqueeze(0).expand(chunk_size, -1, -1)
                else:
                    full_patch_feature_batch = None
                    full_patch_adj_batch = None
            else:
                attack_patch_feature_batch, attack_patch_adj_batch = _generate_attack_patch_batch_for_nodes(
                    trainer=trainer,
                    data=data,
                    target_nodes=chunk_nodes,
                )

                if trigger_eval_enabled:
                    full_patch_feature_batch, full_patch_adj_batch = _generate_full_patch_batch_for_nodes(
                        trainer=trainer,
                        data=data,
                        target_nodes=chunk_nodes,
                        attack_patch_feature_batch=attack_patch_feature_batch,
                        attack_patch_adj_batch=attack_patch_adj_batch,
                    )
                else:
                    full_patch_feature_batch = None
                    full_patch_adj_batch = None

        chunk_attack_counts = _accumulate_prediction_counts_from_batched_graphs(
            trainer=trainer,
            clean_model=clean_model,
            data=data,
            target_nodes=chunk_nodes,
            patch_adj_batch=attack_patch_adj_batch,
            patch_feature_batch=attack_patch_feature_batch,
            clean_edge_indices=clean_edge_indices,
            patch_structure_masks=patch_structure_masks,
            num_classes=num_classes,
            use_full_patch=False,
        )
        attack_pred_counts[start:end] = chunk_attack_counts

        if trigger_eval_enabled:
            chunk_trigger_counts = _accumulate_prediction_counts_from_batched_graphs(
                trainer=trainer,
                clean_model=clean_model,
                data=data,
                target_nodes=chunk_nodes,
                patch_adj_batch=full_patch_adj_batch,
                patch_feature_batch=full_patch_feature_batch,
                clean_edge_indices=clean_edge_indices,
                patch_structure_masks=patch_structure_masks,
                num_classes=num_classes,
                use_full_patch=True,
            )
            trigger_pred_counts[start:end] = chunk_trigger_counts
            del chunk_trigger_counts

        del chunk_attack_counts
        del attack_patch_feature_batch, attack_patch_adj_batch
        if trigger_eval_enabled:
            del full_patch_feature_batch, full_patch_adj_batch

    row_idx = torch.arange(num_eval_nodes, dtype=torch.long)

    clean_majority_pred = clean_pred_counts.argmax(dim=1).long()
    clean_mc_true_rate = (
        clean_pred_counts[row_idx, true_labels_cpu].float() / float(defense_trials)
    )

    attacked_majority_pred = attack_pred_counts.argmax(dim=1).long()
    attacked_mc_true_rate = (
        attack_pred_counts[row_idx, true_labels_cpu].float() / float(defense_trials)
    )
    attacked_mc_misclassification_rate = 1.0 - attacked_mc_true_rate

    if attack_target_class is not None:
        target_class_idx = torch.full(
            (num_eval_nodes,),
            int(attack_target_class),
            dtype=torch.long,
        )
        attacked_mc_target_rate = (
            attack_pred_counts[row_idx, target_class_idx].float() / float(defense_trials)
        )
    else:
        attacked_mc_target_rate = torch.zeros(num_eval_nodes, dtype=torch.float32)

    if trigger_eval_enabled:
        attacked_with_trigger_majority_pred = trigger_pred_counts.argmax(dim=1).long()
        attacked_with_trigger_mc_true_rate = (
            trigger_pred_counts[row_idx, true_labels_cpu].float() / float(defense_trials)
        )
        attacked_with_trigger_mc_misclassification_rate = 1.0 - attacked_with_trigger_mc_true_rate

        if attack_target_class is not None:
            attacked_with_trigger_mc_target_rate = (
                trigger_pred_counts[row_idx, target_class_idx].float() / float(defense_trials)
            )
        else:
            attacked_with_trigger_mc_target_rate = torch.zeros(num_eval_nodes, dtype=torch.float32)
    else:
        attacked_with_trigger_majority_pred = None
        attacked_with_trigger_mc_true_rate = None
        attacked_with_trigger_mc_misclassification_rate = None
        attacked_with_trigger_mc_target_rate = None

    class_stats = {}
    clean_majority_correct = 0
    clean_mc_correct_sum = 0.0

    overall_target_success = 0
    overall_target_success_mc_sum = 0.0
    overall_misclassification = 0
    overall_misclassification_mc_sum = 0.0

    overall_target_success_with_trigger = 0
    overall_target_success_mc_sum_with_trigger = 0.0
    overall_misclassification_with_trigger = 0
    overall_misclassification_mc_sum_with_trigger = 0.0
    overall_clean_recovery_with_trigger = 0
    overall_clean_recovery_mc_sum_with_trigger = 0.0
    overall_attack_suppression_with_trigger = 0
    overall_attack_suppression_mc_sum_with_trigger = 0.0

    for idx, node_idx in enumerate(evaluation_nodes):
        true_label = int(true_labels_cpu[idx].item())

        if true_label not in class_stats:
            class_stats[true_label] = {
                "node_count": 0,
                "clean_majority_correct_count": 0,
                "clean_mc_correct_sum": 0.0,
                "target_success_count": 0,
                "target_success_mc_sum": 0.0,
                "misclassification_count": 0,
                "misclassification_mc_sum": 0.0,
                "target_success_count_with_trigger": 0,
                "target_success_mc_sum_with_trigger": 0.0,
                "misclassification_count_with_trigger": 0,
                "misclassification_mc_sum_with_trigger": 0.0,
                "clean_recovery_count_with_trigger": 0,
                "clean_recovery_mc_sum_with_trigger": 0.0,
                "attack_suppression_count_with_trigger": 0,
                "attack_suppression_mc_sum_with_trigger": 0.0,
            }

        stats = class_stats[true_label]
        stats["node_count"] += 1

        clean_majority_pred_i = int(clean_majority_pred[idx].item())
        clean_mc_true_rate_i = float(clean_mc_true_rate[idx].item())

        if clean_majority_pred_i == true_label:
            clean_majority_correct += 1
            stats["clean_majority_correct_count"] += 1
        clean_mc_correct_sum += clean_mc_true_rate_i
        stats["clean_mc_correct_sum"] += clean_mc_true_rate_i

        attacked_majority_pred_i = int(attacked_majority_pred[idx].item())
        attacked_mc_true_rate_i = float(attacked_mc_true_rate[idx].item())
        attacked_mc_misclassification_rate_i = float(attacked_mc_misclassification_rate[idx].item())

        if attacked_majority_pred_i != true_label:
            overall_misclassification += 1
            stats["misclassification_count"] += 1
        overall_misclassification_mc_sum += attacked_mc_misclassification_rate_i
        stats["misclassification_mc_sum"] += attacked_mc_misclassification_rate_i

        attacked_mc_target_rate_i = float(attacked_mc_target_rate[idx].item())
        if attack_target_class is not None:
            if attacked_majority_pred_i == int(attack_target_class):
                overall_target_success += 1
                stats["target_success_count"] += 1
            overall_target_success_mc_sum += attacked_mc_target_rate_i
            stats["target_success_mc_sum"] += attacked_mc_target_rate_i

        if trigger_eval_enabled:
            attacked_with_trigger_majority_pred_i = int(attacked_with_trigger_majority_pred[idx].item())
            attacked_with_trigger_mc_true_rate_i = float(attacked_with_trigger_mc_true_rate[idx].item())
            attacked_with_trigger_mc_misclassification_rate_i = float(
                attacked_with_trigger_mc_misclassification_rate[idx].item()
            )

            if attacked_with_trigger_majority_pred_i != true_label:
                overall_misclassification_with_trigger += 1
                stats["misclassification_count_with_trigger"] += 1
            overall_misclassification_mc_sum_with_trigger += attacked_with_trigger_mc_misclassification_rate_i
            stats["misclassification_mc_sum_with_trigger"] += attacked_with_trigger_mc_misclassification_rate_i

            if attacked_with_trigger_majority_pred_i == true_label:
                overall_clean_recovery_with_trigger += 1
                stats["clean_recovery_count_with_trigger"] += 1
            overall_clean_recovery_mc_sum_with_trigger += attacked_with_trigger_mc_true_rate_i
            stats["clean_recovery_mc_sum_with_trigger"] += attacked_with_trigger_mc_true_rate_i

            attacked_with_trigger_mc_target_rate_i = float(attacked_with_trigger_mc_target_rate[idx].item())
            if attack_target_class is not None:
                if attacked_with_trigger_majority_pred_i == int(attack_target_class):
                    overall_target_success_with_trigger += 1
                    stats["target_success_count_with_trigger"] += 1
                if attacked_with_trigger_majority_pred_i != int(attack_target_class):
                    overall_attack_suppression_with_trigger += 1
                    stats["attack_suppression_count_with_trigger"] += 1

                overall_target_success_mc_sum_with_trigger += attacked_with_trigger_mc_target_rate_i
                stats["target_success_mc_sum_with_trigger"] += attacked_with_trigger_mc_target_rate_i

                attacked_with_trigger_attack_suppression_rate_mc = 1.0 - attacked_with_trigger_mc_target_rate_i
                overall_attack_suppression_mc_sum_with_trigger += attacked_with_trigger_attack_suppression_rate_mc
                stats["attack_suppression_mc_sum_with_trigger"] += attacked_with_trigger_attack_suppression_rate_mc

    per_class_metrics = {}
    macro_clean_accs = []
    macro_clean_accs_mc = []
    macro_target_rates = []
    macro_target_rates_mc = []
    macro_misclassification_rates = []
    macro_misclassification_rates_mc = []

    macro_target_rates_with_trigger = []
    macro_target_rates_mc_with_trigger = []
    macro_misclassification_rates_with_trigger = []
    macro_misclassification_rates_mc_with_trigger = []
    macro_clean_recovery_rates_with_trigger = []
    macro_clean_recovery_rates_mc_with_trigger = []
    macro_attack_suppression_rates_with_trigger = []
    macro_attack_suppression_rates_mc_with_trigger = []

    for class_label in sorted(class_stats.keys()):
        stats = class_stats[class_label]
        denom = max(1, int(stats["node_count"]))

        clean_acc = _safe_rate(stats["clean_majority_correct_count"], denom)
        clean_acc_mc = _safe_rate(stats["clean_mc_correct_sum"], denom)

        target_rate = _safe_rate(stats["target_success_count"], denom)
        target_rate_mc = _safe_rate(stats["target_success_mc_sum"], denom)
        misclassification_rate = _safe_rate(stats["misclassification_count"], denom)
        misclassification_rate_mc = _safe_rate(stats["misclassification_mc_sum"], denom)

        if trigger_eval_enabled:
            target_rate_with_trigger = _safe_rate(stats["target_success_count_with_trigger"], denom)
            target_rate_mc_with_trigger = _safe_rate(stats["target_success_mc_sum_with_trigger"], denom)
            misclassification_rate_with_trigger = _safe_rate(
                stats["misclassification_count_with_trigger"],
                denom,
            )
            misclassification_rate_mc_with_trigger = _safe_rate(
                stats["misclassification_mc_sum_with_trigger"],
                denom,
            )
            clean_recovery_rate_with_trigger = _safe_rate(
                stats["clean_recovery_count_with_trigger"],
                denom,
            )
            clean_recovery_rate_mc_with_trigger = _safe_rate(
                stats["clean_recovery_mc_sum_with_trigger"],
                denom,
            )
            attack_suppression_rate_with_trigger = _safe_rate(
                stats["attack_suppression_count_with_trigger"],
                denom,
            )
            attack_suppression_rate_mc_with_trigger = _safe_rate(
                stats["attack_suppression_mc_sum_with_trigger"],
                denom,
            )
        else:
            target_rate_with_trigger = None
            target_rate_mc_with_trigger = None
            misclassification_rate_with_trigger = None
            misclassification_rate_mc_with_trigger = None
            clean_recovery_rate_with_trigger = None
            clean_recovery_rate_mc_with_trigger = None
            attack_suppression_rate_with_trigger = None
            attack_suppression_rate_mc_with_trigger = None

        per_class_metrics[str(class_label)] = {
            "node_count": int(stats["node_count"]),
            "clean_accuracy": clean_acc,
            "clean_accuracy_mc": clean_acc_mc,
            "target_success_count": int(stats["target_success_count"]),
            "target_success_rate": target_rate,
            "target_success_rate_mc": target_rate_mc,
            "misclassification_count": int(stats["misclassification_count"]),
            "misclassification_rate": misclassification_rate,
            "misclassification_rate_mc": misclassification_rate_mc,
            "target_success_count_with_trigger": int(stats["target_success_count_with_trigger"]) if trigger_eval_enabled else None,
            "target_success_rate_with_trigger": target_rate_with_trigger,
            "target_success_rate_mc_with_trigger": target_rate_mc_with_trigger,
            "misclassification_count_with_trigger": int(stats["misclassification_count_with_trigger"]) if trigger_eval_enabled else None,
            "misclassification_rate_with_trigger": misclassification_rate_with_trigger,
            "misclassification_rate_mc_with_trigger": misclassification_rate_mc_with_trigger,
            "clean_recovery_count_with_trigger": int(stats["clean_recovery_count_with_trigger"]) if trigger_eval_enabled else None,
            "clean_recovery_rate_with_trigger": clean_recovery_rate_with_trigger,
            "clean_recovery_rate_mc_with_trigger": clean_recovery_rate_mc_with_trigger,
            "attack_suppression_count_with_trigger": int(stats["attack_suppression_count_with_trigger"]) if trigger_eval_enabled else None,
            "attack_suppression_rate_with_trigger": attack_suppression_rate_with_trigger,
            "attack_suppression_rate_mc_with_trigger": attack_suppression_rate_mc_with_trigger,
        }

        macro_clean_accs.append(clean_acc)
        macro_clean_accs_mc.append(clean_acc_mc)
        macro_target_rates.append(target_rate)
        macro_target_rates_mc.append(target_rate_mc)
        macro_misclassification_rates.append(misclassification_rate)
        macro_misclassification_rates_mc.append(misclassification_rate_mc)

        if trigger_eval_enabled:
            macro_target_rates_with_trigger.append(target_rate_with_trigger)
            macro_target_rates_mc_with_trigger.append(target_rate_mc_with_trigger)
            macro_misclassification_rates_with_trigger.append(misclassification_rate_with_trigger)
            macro_misclassification_rates_mc_with_trigger.append(misclassification_rate_mc_with_trigger)
            macro_clean_recovery_rates_with_trigger.append(clean_recovery_rate_with_trigger)
            macro_clean_recovery_rates_mc_with_trigger.append(clean_recovery_rate_mc_with_trigger)
            macro_attack_suppression_rates_with_trigger.append(attack_suppression_rate_with_trigger)
            macro_attack_suppression_rates_mc_with_trigger.append(attack_suppression_rate_mc_with_trigger)

    total_nodes = max(1, len(evaluation_nodes))
    result = {
        "evaluation_mode": "shared_generator_cached_global_fullgraph_randomized_defense",
        "defense_cache_mode": "pre_sampled_global_full_graph",
        "shared_generator": bool(getattr(trainer, "shared_generator", True)),
        "generator_subgraph_hops": int(
            _pick_first_not_none(trainer, ["generator_subgraph_hops"], default=1)
        ),
        "used_fixed_trained_patch": bool(patch_mode == "fixed"),
        "patch_mode": str(patch_mode),
        "trigger_evaluation_enabled": bool(trigger_eval_enabled),
        "num_trigger_nodes": int(getattr(trainer, "num_trigger_nodes", 0)),
        "defense_enabled": True,
        "defense_trials": int(defense_trials),
        "defense_drop_prob": float(drop_prob),
        "defense_num_hops": int(num_hops),
        "defense_scope": "global_full_graph_and_patch_mask",
        "evaluation_batch_size": int(eval_batch_size),
        "evaluation_node_indices": [int(x) for x in evaluation_nodes],
        "evaluation_node_count": int(len(evaluation_nodes)),
        "attack_target_class": attack_target_class,
        "clean_accuracy_on_evaluation_nodes": _safe_rate(clean_majority_correct, total_nodes),
        "clean_accuracy_on_evaluation_nodes_mc": _safe_rate(clean_mc_correct_sum, total_nodes),
        "overall_target_success_count": int(overall_target_success),
        "overall_target_success_rate": _safe_rate(overall_target_success, total_nodes),
        "overall_target_success_rate_mc": _safe_rate(overall_target_success_mc_sum, total_nodes),
        "overall_misclassification_count": int(overall_misclassification),
        "overall_misclassification_rate": _safe_rate(overall_misclassification, total_nodes),
        "overall_misclassification_rate_mc": _safe_rate(overall_misclassification_mc_sum, total_nodes),
        "macro_avg_clean_accuracy": _safe_rate(sum(macro_clean_accs), len(macro_clean_accs)),
        "macro_avg_clean_accuracy_mc": _safe_rate(sum(macro_clean_accs_mc), len(macro_clean_accs_mc)),
        "macro_avg_target_success_rate": _safe_rate(sum(macro_target_rates), len(macro_target_rates)),
        "macro_avg_target_success_rate_mc": _safe_rate(sum(macro_target_rates_mc), len(macro_target_rates_mc)),
        "macro_avg_misclassification_rate": _safe_rate(sum(macro_misclassification_rates), len(macro_misclassification_rates)),
        "macro_avg_misclassification_rate_mc": _safe_rate(
            sum(macro_misclassification_rates_mc),
            len(macro_misclassification_rates_mc),
        ),
        "overall_target_success_count_with_trigger": int(overall_target_success_with_trigger) if trigger_eval_enabled else None,
        "overall_target_success_rate_with_trigger": _safe_rate(overall_target_success_with_trigger, total_nodes) if trigger_eval_enabled else None,
        "overall_target_success_rate_mc_with_trigger": _safe_rate(overall_target_success_mc_sum_with_trigger, total_nodes) if trigger_eval_enabled else None,
        "overall_misclassification_count_with_trigger": int(overall_misclassification_with_trigger) if trigger_eval_enabled else None,
        "overall_misclassification_rate_with_trigger": _safe_rate(overall_misclassification_with_trigger, total_nodes) if trigger_eval_enabled else None,
        "overall_misclassification_rate_mc_with_trigger": _safe_rate(
            overall_misclassification_mc_sum_with_trigger,
            total_nodes,
        ) if trigger_eval_enabled else None,
        "overall_clean_recovery_count_with_trigger": int(overall_clean_recovery_with_trigger) if trigger_eval_enabled else None,
        "overall_clean_recovery_rate_with_trigger": _safe_rate(overall_clean_recovery_with_trigger, total_nodes) if trigger_eval_enabled else None,
        "overall_clean_recovery_rate_mc_with_trigger": _safe_rate(
            overall_clean_recovery_mc_sum_with_trigger,
            total_nodes,
        ) if trigger_eval_enabled else None,
        "overall_attack_suppression_count_with_trigger": int(overall_attack_suppression_with_trigger) if trigger_eval_enabled else None,
        "overall_attack_suppression_rate_with_trigger": _safe_rate(
            overall_attack_suppression_with_trigger,
            total_nodes,
        ) if trigger_eval_enabled else None,
        "overall_attack_suppression_rate_mc_with_trigger": _safe_rate(
            overall_attack_suppression_mc_sum_with_trigger,
            total_nodes,
        ) if trigger_eval_enabled else None,
        "macro_avg_target_success_rate_with_trigger": _safe_rate(
            sum(macro_target_rates_with_trigger),
            len(macro_target_rates_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_target_success_rate_mc_with_trigger": _safe_rate(
            sum(macro_target_rates_mc_with_trigger),
            len(macro_target_rates_mc_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_misclassification_rate_with_trigger": _safe_rate(
            sum(macro_misclassification_rates_with_trigger),
            len(macro_misclassification_rates_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_misclassification_rate_mc_with_trigger": _safe_rate(
            sum(macro_misclassification_rates_mc_with_trigger),
            len(macro_misclassification_rates_mc_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_clean_recovery_rate_with_trigger": _safe_rate(
            sum(macro_clean_recovery_rates_with_trigger),
            len(macro_clean_recovery_rates_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_clean_recovery_rate_mc_with_trigger": _safe_rate(
            sum(macro_clean_recovery_rates_mc_with_trigger),
            len(macro_clean_recovery_rates_mc_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_attack_suppression_rate_with_trigger": _safe_rate(
            sum(macro_attack_suppression_rates_with_trigger),
            len(macro_attack_suppression_rates_with_trigger),
        ) if trigger_eval_enabled else None,
        "macro_avg_attack_suppression_rate_mc_with_trigger": _safe_rate(
            sum(macro_attack_suppression_rates_mc_with_trigger),
            len(macro_attack_suppression_rates_mc_with_trigger),
        ) if trigger_eval_enabled else None,
        "per_class_metrics": per_class_metrics,
    }

    print(
        "[Eval Summary] "
        f"clean_acc={result['clean_accuracy_on_evaluation_nodes']:.4f} "
        f"target_success={result['overall_target_success_rate']:.4f} "
        f"misclf={result['overall_misclassification_rate']:.4f} "
        f"trigger_recovery={result['overall_clean_recovery_rate_with_trigger'] if trigger_eval_enabled else 'N/A'}"
    )
    return result
