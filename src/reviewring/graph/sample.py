"""Relational neighbourhood sampling for mini-batch GNN training.

Implements the spec's per-relation fanout (``[10, 5]``) with a deterministic
seed so evaluation sampling is reproducible. Sampling uses per-relation CSR
adjacency (symmetrised for message passing) and expands a batch of seed nodes
layer by layer. Sampling is fully vectorised (segment-wise top-k by random
keys, i.e. uniform without replacement), so the 7.7M-edge YelpChi union is
trainable on CPU. Pure torch/numpy: no PyG compiled extensions required,
which keeps Colab installs robust (the spec warns about PyG version matching).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class LayerAdjacency:
    """One sampled hop for a batch, localised.

    Local ids: the layer's ``seeds`` (dst side) are ids ``0..n_seeds-1`` in the
    order given; sampled frontier nodes (src side) follow. The conv computes
    output embeddings for the seed block.
    """

    local_edge_index: dict[str, np.ndarray] = field(default_factory=dict)
    all_local_nodes: np.ndarray = None  # global ids of seeds+frontier, in local order
    n_seeds: int = 0
    n_local: int = 0


class RelationalNeighborSampler:
    """Precomputes per-relation symmetrised CSR structures for sampling."""

    def __init__(self, relations: dict[str, np.ndarray], n_nodes: int, seed: int = 0):
        self.n_nodes = n_nodes
        self.rng = np.random.default_rng(seed)
        self.relation_names = list(relations.keys())
        self.csr: dict[str, sp.csr_matrix] = {}
        for name, edges in relations.items():
            rows = np.concatenate([edges[0], edges[1]])
            cols = np.concatenate([edges[1], edges[0]])
            data = np.ones(len(rows), dtype=np.int8)
            mat = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
            mat.data[:] = 1
            mat.sort_indices()
            self.csr[name] = mat
        self._local_map = np.full(n_nodes, -1, dtype=np.int64)

    def _sample_edges(self, dst_nodes: np.ndarray, fanout: int, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Vectorised neighbour sampling, following the original GraphSAGE
        scheme: up to ``fanout`` neighbours per node per relation drawn
        uniformly *with replacement* (duplicates then removed). Exact
        uniform-without-replacement over millions of slots would require a
        full per-node scan; the with-replacement draw is the standard,
        documented approximation and is unbiased per draw.

        Returns per relation: (src_global, dst_global).
        """
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, mat in self.csr.items():
            indptr, indices = mat.indptr, mat.indices
            start = indptr[dst_nodes]
            deg = (indptr[dst_nodes + 1] - start).astype(np.int64)
            has_edge = deg > 0
            if not has_edge.any():
                out[name] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
                continue
            d = dst_nodes[has_edge]
            deg_d = deg[has_edge]
            start_d = start[has_edge]
            # nodes with degree <= fanout keep ALL their neighbours (exact);
            # higher-degree nodes draw `fanout` with replacement + dedup
            full = deg_d <= fanout
            full_idx = np.where(full)[0]
            part_idx = np.where(~full)[0]
            src_parts = []
            dst_parts = []
            if len(full_idx):
                seg_full = np.repeat(full_idx, deg_d[full_idx])
                # expand each node's full CSR slice
                cum = np.cumsum(deg_d[full_idx])
                pos_full = np.repeat(start_d[full_idx], deg_d[full_idx]) + (
                    np.arange(int(deg_d[full_idx].sum())) - np.repeat(cum - deg_d[full_idx], deg_d[full_idx])
                )
                src_parts.append(indices[pos_full])
                dst_parts.append(d[seg_full])
            if len(part_idx):
                pick = np.full(len(part_idx), fanout, dtype=np.int64)
                total = int(pick.sum())
                seg = np.repeat(part_idx, pick)
                pos = np.repeat(start_d[part_idx], pick) + (
                    rng.random(total) * np.repeat(deg_d[part_idx].astype(np.float64), pick)
                ).astype(np.int64)
                src_parts.append(indices[pos])
                dst_parts.append(d[seg])
            src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int64)
            dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=np.int64)
            # deduplicate (src, dst) pairs, preserving no particular order
            pair = src.astype(np.int64) * self.n_nodes + dst.astype(np.int64)
            pair = np.unique(pair)
            src = (pair // self.n_nodes).astype(np.int64)
            dst = (pair % self.n_nodes).astype(np.int64)
            out[name] = (src, dst)
        return out

    def sample_layer(self, seed_nodes: np.ndarray, fanout: int, rng: np.random.Generator) -> LayerAdjacency:
        seeds = np.asarray(seed_nodes, dtype=np.int64)
        per_rel = self._sample_edges(seeds, fanout, rng)
        srcs = [s for s, _ in per_rel.values() if len(s)]
        if srcs:
            all_src = np.unique(np.concatenate(srcs))
        else:
            all_src = np.zeros(0, dtype=np.int64)
        # local mapping: seeds keep ids 0..n_seeds-1; *new* frontier after
        self._local_map[seeds] = np.arange(len(seeds))
        frontier_mask = self._local_map[all_src] < 0
        frontier = all_src[frontier_mask]
        self._local_map[frontier] = len(seeds) + np.arange(len(frontier))
        n_local = len(seeds) + len(frontier)
        local_edges: dict[str, np.ndarray] = {}
        for name, (src, dst) in per_rel.items():
            src_local = self._local_map[src]
            dst_local = self._local_map[dst]
            local_edges[name] = np.vstack([src_local, dst_local])
        all_nodes = np.concatenate([seeds, frontier]) if len(frontier) else seeds
        # reset map entries for reuse
        self._local_map[seeds] = -1
        if len(frontier):
            self._local_map[frontier] = -1
        return LayerAdjacency(
            local_edge_index=local_edges,
            all_local_nodes=all_nodes,
            n_seeds=len(seeds),
            n_local=n_local,
        )


def build_layer_stack(
    sampler: RelationalNeighborSampler,
    seed_nodes: np.ndarray,
    fanouts: list[int],
    rng: np.random.Generator,
) -> list[LayerAdjacency]:
    """Sample hop by hop; returns layers ordered last-to-first (PyG style).

    The conv loop consumes layers in list order; between layers the feature
    rows for the next layer's input nodes are the previous layer's output rows
    restricted to its seed block (alignment holds by construction).
    """
    layers = []
    current = np.asarray(seed_nodes, dtype=np.int64)
    for fanout in fanouts:
        adjacency = sampler.sample_layer(current, fanout, rng)
        current = adjacency.all_local_nodes
        layers.append(adjacency)
    layers.reverse()
    return layers
