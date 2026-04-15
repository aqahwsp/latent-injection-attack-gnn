import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops, subgraph, to_undirected
from utils import ensure_masks
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.0
DEFAULT_TEST_RATIO = 0.3


def load_data(path):
    with np.load(path, allow_pickle=True) as graph_file:
        graph = {k: np.array(graph_file[k], copy=True) for k in graph_file.files}

    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    X = sp.csr_matrix((np.ones(graph['X'].shape[1]), graph['X']), dtype=np.float32).todense()
    y = graph['y']
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc, graph



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


def sanitize_edge_index_binary(edge_index, num_nodes):
    """
    统一把图处理成：
    1. 无向图
    2. 去除自环
    3. 边只保留 0/1 两种状态（兼容不同 PyG 版本的 coalesce 返回值）
    """
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0), dtype=torch.long)
    edge_index = edge_index.long()
    edge_index = _to_undirected_edge_index_only(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = _coalesce_edge_index_only(edge_index, num_nodes)
    return edge_index.long()


def validate_data_consistency(data, name="data"):
    if not hasattr(data, "x") or data.x is None:
        raise ValueError(f"{name}.x 不能为空。")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError(f"{name}.edge_index 不能为空。")

    num_nodes = int(data.x.size(0))
    if getattr(data, "num_nodes", None) is not None:
        data.num_nodes = int(num_nodes)

    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError(f"{name}.edge_index 形状非法，期望 [2, E]，实际为 {tuple(data.edge_index.shape)}。")

    if data.edge_index.numel() > 0:
        min_node_idx = int(data.edge_index.min().item())
        max_node_idx = int(data.edge_index.max().item())
        if min_node_idx < 0 or max_node_idx >= num_nodes:
            raise ValueError(
                f"{name} 的 edge_index 节点编号越界：最小值 {min_node_idx}，最大值 {max_node_idx}，"
                f"但 x 只有 {num_nodes} 个节点。请使用与节点子集一致的子图 edge_index。"
            )

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, mask_name, None)
        if mask is not None and int(mask.numel()) != num_nodes:
            raise ValueError(
                f"{name}.{mask_name} 长度为 {int(mask.numel())}，但节点数为 {num_nodes}，两者不一致。"
            )

    return data


def csr_to_edge_index(A_csr):
    row, col = A_csr.nonzero()
    edge_index = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)
    return sanitize_edge_index_binary(edge_index, A_csr.shape[0])


def create_node_masks(
    num_nodes,
    labels=None,
    target_class=None,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=0,
):
    """
    按比例随机划分 train/val/test。
    这里不再使用旧版 0.6/0.2/0.2，也不在这里排除 target_class；
    目标类过滤交给下游训练节点选择逻辑处理。
    """
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train/val/test 比例之和必须为 1，当前为 {total_ratio}。"
        )

    num_nodes = int(num_nodes)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(num_nodes * float(train_ratio))
    n_val = int(num_nodes * float(val_ratio))
    n_test = num_nodes - n_train - n_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    if n_val > 0:
        val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:n_train + n_val + n_test]] = True

    return train_mask, val_mask, test_mask



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


def extract_masks_from_graph(graph, num_nodes):
    candidate_triplets = [
        ('train_mask', 'val_mask', 'test_mask'),
        ('train_masks', 'val_masks', 'test_masks'),
        ('idx_train', 'idx_val', 'idx_test'),
        ('train_idx', 'val_idx', 'test_idx'),
        ('train_indices', 'val_indices', 'test_indices'),
    ]

    for train_key, val_key, test_key in candidate_triplets:
        if train_key in graph and val_key in graph and test_key in graph:
            train_mask = _mask_from_graph_value(graph[train_key], num_nodes)
            val_mask = _mask_from_graph_value(graph[val_key], num_nodes)
            test_mask = _mask_from_graph_value(graph[test_key], num_nodes)
            return train_mask, val_mask, test_mask
    return None, None, None


def build_pyg_data(
    npz_path,
    out_file=None,
    target_class=None,
    train_mask_exists=False,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=0,
):
    A, X, y, n, d, nc, graph = load_data(npz_path)
    edge_index = csr_to_edge_index(A)

    train_mask, val_mask, test_mask = extract_masks_from_graph(graph, n)
    if train_mask is None or val_mask is None or test_mask is None:
        train_mask, val_mask, test_mask = create_node_masks(
            num_nodes=n,
            labels=torch.as_tensor(y),
            target_class=target_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    data = Data(
        x=torch.as_tensor(np.asarray(X), dtype=torch.float32),
        y=torch.as_tensor(np.asarray(y), dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=int(n),
        num_features=int(d),
        num_classes=int(nc),
    )
    return validate_data_consistency(data, name="build_pyg_data")



def normalize_features(data):
    transform = T.Compose([T.NormalizeFeatures()])
    data = transform(data)
    return data


def convert_to_undirected(data):
    validate_data_consistency(data, name="data_before_convert_to_undirected")
    if data.edge_index is not None:
        data.edge_index = sanitize_edge_index_binary(data.edge_index, data.x.size(0))
    return validate_data_consistency(data, name="data_after_convert_to_undirected")


def get_masked_subgraph(data, node_mask, active_mask_name="train_mask"):
    validate_data_consistency(data, name="data")

    node_mask = torch.as_tensor(node_mask, dtype=torch.bool, device=data.x.device).view(-1)
    if int(node_mask.numel()) != int(data.x.size(0)):
        raise ValueError(
            f"node_mask 长度为 {int(node_mask.numel())}，但 data.x 对应的节点数为 {int(data.x.size(0))}。"
        )

    selected_nodes = node_mask.nonzero(as_tuple=False).view(-1)
    if selected_nodes.numel() == 0:
        raise ValueError("node_mask 为空，无法构造子图。")

    edge_attr = getattr(data, "edge_attr", None)
    selected_nodes_for_edges = selected_nodes.to(data.edge_index.device)
    sub_edge_index, sub_edge_attr = subgraph(
        selected_nodes_for_edges,
        data.edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
    )
    sub_edge_index = sanitize_edge_index_binary(sub_edge_index, len(selected_nodes))

    subgraph_data = data.clone()
    subgraph_data.num_nodes = int(selected_nodes.numel())
    subgraph_data.x = data.x[selected_nodes]
    subgraph_data.y = data.y[selected_nodes]
    subgraph_data.edge_index = sub_edge_index
    if sub_edge_attr is not None:
        subgraph_data.edge_attr = sub_edge_attr
    subgraph_data.original_node_idx = selected_nodes.detach().cpu()

    mask_device = subgraph_data.x.device
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if mask_name == active_mask_name:
            setattr(subgraph_data, mask_name, torch.ones(subgraph_data.num_nodes, dtype=torch.bool, device=mask_device))
        else:
            setattr(subgraph_data, mask_name, torch.zeros(subgraph_data.num_nodes, dtype=torch.bool, device=mask_device))

    subgraph_data.num_features = int(subgraph_data.x.size(1))
    if hasattr(data, "num_classes") and data.num_classes is not None:
        subgraph_data.num_classes = int(data.num_classes)

    validate_data_consistency(subgraph_data, name=f"{active_mask_name}_subgraph")
    return subgraph_data, selected_nodes


def get_train_subgraph(data, train_mask):
    return get_masked_subgraph(data, train_mask, active_mask_name="train_mask")


def get_val_subgraph(data, val_mask):
    return get_masked_subgraph(data, val_mask, active_mask_name="val_mask")


def get_test_subgraph(data, test_mask):
    return get_masked_subgraph(data, test_mask, active_mask_name="test_mask")


def get_split_subgraphs(data):
    validate_data_consistency(data, name="data")
    return {
        "train": get_train_subgraph(data, data.train_mask),
        "val": get_val_subgraph(data, data.val_mask),
        "test": get_test_subgraph(data, data.test_mask),
    }


def preprocess_data(
    data,
    train_mask_exists=False,
    target_class=None,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=0,
):
    print("开始数据预处理...")
    data = normalize_features(data)
    print("节点特征已归一化。")

    data = convert_to_undirected(data)
    print("图已转换为无向图，且边仅保留 0/1 状态。")

    if not train_mask_exists:
        print("为训练集、验证集和测试集创建新的节点掩码...")
        train_mask, val_mask, test_mask = create_node_masks(
            num_nodes=data.num_nodes,
            labels=data.y,
            target_class=target_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        print("节点掩码已创建。")
    else:
        print("假定数据中已包含有效的训练、验证和测试掩码。")

    validate_data_consistency(data, name="preprocessed_data")
    print("数据预处理完成。")
    return data



if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./tmp/Cora', name='Cora')
    data = dataset[0]

    processed_data = preprocess_data(data.clone(), train_mask_exists=False)
    print("\n处理后的数据摘要:")
    print(processed_data)
    print(f"训练集掩码数量: {processed_data.train_mask.sum().item()}")
    print(f"验证集掩码数量: {processed_data.val_mask.sum().item()}")
    print(f"测试集掩码数量: {processed_data.test_mask.sum().item()}")
