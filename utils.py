import numpy as np
import torch
import torch.nn as nn
import os
import scipy.sparse as sp
from torch_geometric.utils import add_remaining_self_loops,to_dense_adj
import torch_geometric.transforms as T
import gc

def init_random_seed(SEED=2021):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
init_random_seed()

def load_data(path):
    with np.load(path, allow_pickle=True) as graph_file:
        graph = {k: np.array(graph_file[k], copy=True) for k in graph_file.files}

    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    data = (np.ones(graph['X'].shape[1]), graph['X'])
    X = sp.csr_matrix(data, dtype=np.float32).todense()
    y = graph['y']
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc


def get_degrees(edge_index):
    adj_dense = torch.squeeze(to_dense_adj(edge_index))
    adj_dense.fill_diagonal_(0)
    (adj_dense==adj_dense.T).all()
    degrees = adj_dense.sum(0).cpu().numpy().astype(np.int16)
    return degrees

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split(
    labels,
    n_per_class=20,
    seed=0,
    force_ratio=True,
    train_ratio=0.7,
    val_ratio=0.0,
):
    """
    默认按比例随机划分。
    只有当 force_ratio=False 时，才启用旧的“每类固定 n_per_class 个点”的老逻辑。
    """
    labels = np.asarray(labels)
    num_nodes = len(labels)

    if not force_ratio:
        np.random.seed(seed)
        nc = labels.max() + 1
        split_train, split_val = [], []

        for l in range(nc):
            perm = np.random.RandomState(seed=seed + int(l)).permutation(
                (labels == l).nonzero()[0]
            )
            split_train.append(perm[:n_per_class])
            split_val.append(perm[n_per_class:2 * n_per_class])

        split_train = np.random.RandomState(seed=seed).permutation(
            np.concatenate(split_train)
        )
        split_val = np.random.RandomState(seed=seed + 1).permutation(
            np.concatenate(split_val)
        )
        split_test = np.setdiff1d(
            np.arange(len(labels)),
            np.concatenate((split_train, split_val))
        )

        print("Split mode: fixed n_per_class")
        print("Number of samples per class:", n_per_class)
        print(
            "Training-validation-testing Size:",
            len(split_train), len(split_val), len(split_test)
        )
        return split_train, split_val, split_test

    if not (0 < float(train_ratio) < 1):
        raise ValueError("train_ratio 必须在 (0, 1) 范围内。")
    if not (0 <= float(val_ratio) < 1):
        raise ValueError("val_ratio 必须在 [0, 1) 范围内。")
    if float(train_ratio) + float(val_ratio) >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1。")

    perm = np.random.RandomState(seed=seed).permutation(np.arange(num_nodes))

    n_train = int(num_nodes * float(train_ratio))
    n_val = int(num_nodes * float(val_ratio))

    split_train = perm[:n_train]
    split_val = perm[n_train:n_train + n_val]
    split_test = perm[n_train + n_val:]

    print("Split mode: ratio")
    print(
        "Training-validation-testing Size:",
        len(split_train), len(split_val), len(split_test)
    )
    return split_train, split_val, split_test



def normalize(adj):
    degree = torch.sum(adj,dim=0)
    D_half_norm = torch.pow(degree, -0.5)
    D_half_norm = torch.nan_to_num(D_half_norm, nan=0.0, posinf=0.0, neginf=0.0)
    D_half_norm = torch.diag(D_half_norm)
    DAD = torch.mm(torch.mm(D_half_norm,adj), D_half_norm)
    return DAD

def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

def listSubset(A_list,index_list):
    '''take out the elements of a list (A_list) by a index list (index_list)'''
    return [A_list[i] for i in index_list]

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.0
DEFAULT_TEST_RATIO = 0.3


def normalize_features(data):
    if getattr(data, "x", None) is None:
        return data
    data.x = data.x.float()
    data = T.NormalizeFeatures()(data)
    return data
def create_node_masks(
    num_nodes,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=42,
    device=None
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    n_test = num_nodes - n_train - n_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    if device is not None:
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    return train_mask, val_mask, test_mask
def ensure_masks(
    data,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=42
):
    need_new_masks = (
        not hasattr(data, "train_mask") or data.train_mask is None or
        not hasattr(data, "val_mask") or data.val_mask is None or
        not hasattr(data, "test_mask") or data.test_mask is None
    )

    if need_new_masks:
        device = data.x.device if getattr(data, "x", None) is not None else None
        train_mask, val_mask, test_mask = create_node_masks(
            num_nodes=data.num_nodes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            device=device
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    return data

def cleanup_memory(verbose=False):
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"[CUDA Memory] allocated={allocated:.2f} MB, reserved={reserved:.2f} MB")
