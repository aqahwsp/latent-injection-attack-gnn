import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from torch_geometric.nn import APPNP, GATConv, GATv2Conv, GCNConv
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

from utils import count_arr


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_remaining_self_loops(edge_index)
        x = self.conv1(features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SmoothGCN(GCN):
    """Multi-smoothing model for GCN."""

    def __init__(self, in_channels, out_channels, hidden_channels, dropout, config, device):
        super().__init__(in_channels, out_channels, hidden_channels, dropout)
        self.config = config
        self.device = device
        self.nclass = out_channels
        self.p_e = torch.tensor(float(config["p_e"]), device=device)
        self.p_n = torch.tensor(float(config["p_n"]), device=device)

    def perturbation(self, adj_dense: Tensor) -> Tensor:
        adj_dense = adj_dense.to(self.device)
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0

        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = (adj_triu == 1) * torch.bernoulli(
            torch.ones(size, device=self.device) * (1 - self.p_e)
        )
        node_keep = torch.bernoulli(torch.ones(size[0], device=self.device) * (1 - self.p_n))
        adj_triu = adj_triu.mul(node_keep.unsqueeze(0))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor) -> Tensor:
        with torch.no_grad():
            adj_dense = to_dense_adj(edge_index, max_num_nodes=int(features.size(0))).squeeze(0)
            adj_dense = self.perturbation(adj_dense)
            edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features: Tensor, edge_index: Tensor, num: int):
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for _ in tqdm(range(int(num)), desc="ProcessingMonteCarlo"):
            predictions = self.forward_perturb(features, edge_index).argmax(1)
            counts += count_arr(predictions.detach().cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2]
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = float(dropout)
        self.heads = int(heads)

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
        )
        self.conv2 = GATConv(
            hidden_channels * self.heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=self.dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SmoothGAT(GAT):
    def __init__(self, in_channels, out_channels, hidden_channels, heads, dropout, config, device):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )
        self.config = config
        self.device = device
        self.nclass = out_channels
        self.p_e = torch.tensor(float(config["p_e"]), device=device)
        self.p_n = torch.tensor(float(config["p_n"]), device=device)

    def perturbation(self, adj_dense: Tensor) -> Tensor:
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0

        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = (adj_triu == 1) * torch.bernoulli(
            torch.ones(size, device=self.device) * (1 - self.p_e)
        )
        node_keep = torch.bernoulli(torch.ones(size[0], device=self.device) * (1 - self.p_n))
        adj_triu = adj_triu.mul(node_keep.unsqueeze(0))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor) -> Tensor:
        with torch.no_grad():
            adj_dense = to_dense_adj(edge_index, max_num_nodes=int(features.size(0))).squeeze(0)
            adj_dense = self.perturbation(adj_dense)
            edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features: Tensor, edge_index: Tensor, num: int):
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for _ in tqdm(range(int(num)), desc="ProcessingMonteCarlo"):
            predictions = self.forward_perturb(features, edge_index).argmax(1)
            counts += count_arr(predictions.detach().cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2]
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2
