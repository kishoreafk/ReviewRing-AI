"""Expert networks (spec 11.2): text MLP, timing MLP, relational GraphSAGE.

All expert representations are size ``hidden_dim`` (default 64). The graph
expert uses separate convolutions per relation, aggregates their outputs by
mean, and retains a self-feature path. A single-relation variant is available
for ablations. Loss heads are scalar logits trained with BCEWithLogitsLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextExpert(nn.Module):
    """384 (or fallback dim) -> 128 -> 64 MLP with a scalar logit head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (repr [B, H], logit [B])."""
        h = self.encoder(x)
        return h, self.head(h).squeeze(-1)


class TimeExpert(nn.Module):
    """F_time -> 64 -> 64 MLP with missingness indicators concatenated."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # indicators are concatenated to the imputed feature vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, missing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, missing.float()], dim=-1))
        return h, self.head(h).squeeze(-1)


class RelationalSageConv(nn.Module):
    """One relation-aware GraphSAGE stage over sampled subgraphs.

    For each relation r: m_r = mean of x_src over sampled edges, h_r =
    W_r[m_r] ; self path: h_self = W_self[x_dst]. Output = ReLU(mean_r h_r +
    h_self). Layers receive LayerAdjacency-like dicts with local edge indices
    and compute outputs for the seed block (ids 0..n_seeds-1).
    """

    def __init__(self, in_dim: int, out_dim: int, relations: list[str], dropout: float = 0.2):
        super().__init__()
        self.relations = list(relations)
        self.rel_lin = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in self.relations})
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: dict[str, torch.Tensor],
        n_seeds: int,
    ) -> torch.Tensor:
        """x: [n_local, in]; edge_index: relation -> [2, E] local (src, dst).

        Returns [n_seeds, out] embeddings for the seed (dst) block.
        """
        n_local = x.shape[0]
        dst = torch.arange(n_seeds, device=x.device)
        rel_outputs = []
        for rel in self.relations:
            if rel not in edge_index or edge_index[rel].numel() == 0:
                continue
            src, dst_e = edge_index[rel][0], edge_index[rel][1]
            # mean aggregation over sampled in-edges per dst node
            agg = torch.zeros((n_local, x.shape[1]), dtype=x.dtype, device=x.device)
            msgs = x[src]
            agg.index_add_(0, dst_e, msgs)
            deg = torch.zeros(n_local, dtype=x.dtype, device=x.device)
            deg.index_add_(0, dst_e, torch.ones(len(dst_e), dtype=x.dtype, device=x.device))
            mean = agg / deg.clamp(min=1.0).unsqueeze(-1)
            rel_outputs.append(self.rel_lin[rel](mean[dst]))
        if rel_outputs:
            out = torch.stack(rel_outputs, dim=0).mean(dim=0)
        else:
            out = torch.zeros((n_seeds, self.self_lin.out_features), dtype=x.dtype, device=x.device)
        out = out + self.self_lin(x[dst])
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class GraphExpert(nn.Module):
    """Two relation-aware GraphSAGE stages with 64 hidden units.

    ``sampled=True`` mode consumes layer-wise sampled subgraphs (list of
    (edge_dict, n_seeds, n_local)) ordered last-to-first. ``sampled=False``
    mode runs full-batch on a complete graph (Track B sizes) with the same
    parameter shapes.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        relations: list[str] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        relations = relations or ["rel0"]
        convs = []
        dims = [in_dim] + [hidden_dim] * layers
        for i in range(layers):
            convs.append(RelationalSageConv(dims[i], dims[i + 1], relations, dropout=dropout))
        self.convs = nn.ModuleList(convs)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout


    def forward_full(
        self, x: torch.Tensor, edge_index: dict[str, torch.Tensor], n_seeds: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0] if n_seeds is None else n_seeds
        for conv in self.convs:
            x = conv(x, edge_index, n)
        return x, self.head(x).squeeze(-1)
