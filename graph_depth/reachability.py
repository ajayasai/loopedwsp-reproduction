"""Analytic directed reachability with matched stored-coefficient budgets.

adj[b,u,v] denotes a directed edge u -> v. Every model receives the same graph,
vertex coordinates and endpoint indices. The local operators transport a
source bit. The global operators carry a reachability row at each vertex and
compose the relation using standard dot-product softmax attention.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


BUDGETS = (52, 100)


def _coefficient(value):
    return nn.Parameter(torch.as_tensor(value, dtype=torch.float64), requires_grad=False)


class _Stage(nn.Module):
    def __init__(self, nodes, heads, global_attention):
        super().__init__()
        self.nodes = nodes
        self.head_count = heads
        self.global_attention = global_attention
        # Group fixed vertex indices by residue. Padding introduces no real key.
        ix = torch.arange(heads * math.ceil(nodes / heads)).reshape(-1, heads).T
        self.register_buffer("live", ix < nodes, persistent=False)
        self.register_buffer("indices", ix.clamp(max=nodes-1), persistent=False)
        group_size = (ix < nodes).sum(-1).to(torch.float64)
        if global_attention:
            # Scores are 2048 for an active intermediary, 1024 for the null
            # token, and 0 for an inactive key before stable softmax.
            q = torch.full((heads,), 2048 * math.sqrt(nodes+1), dtype=torch.float64)
            gains = 2 * group_size
        else:
            # Even one active predecessor gets mass > 2/3 in the presence
            # of all inactive keys in its shard and a null token.
            q = (4 * (group_size+1)).log()
            gains = torch.full((heads,), 2., dtype=torch.float64)
        self.q = _coefficient(q)
        self.k = _coefficient(torch.ones_like(q))
        self.v = _coefficient(gains)
        self.o = _coefficient(torch.ones_like(q))
        # x - ReLU(x - 1): zero for x=0 and exactly one for x>=1.
        # The affine residual MLP includes both biases in the budget.
        self.w1 = _coefficient(1.)
        self.b1 = _coefficient(-1.)
        self.w2 = _coefficient(-1.)
        self.b2 = _coefficient(0.)

    def prepare(self, relation, dtype):
        if self.global_attention:
            # Deterministic input coordinates, computed from public vertex IDs.
            # Projections share scalar coefficients across these coordinates.
            coordinates = F.one_hot(self.indices, num_classes=self.nodes).to(dtype)
            real_keys = F.pad(coordinates, (0, 1))
            null_key = real_keys.new_zeros((self.head_count, 1, self.nodes+1))
            null_key[:, :, -1] = 0.5
            return torch.cat((real_keys, null_key), dim=1)
        # Return [batch, receiver, head, source slot]. Every supplied edge
        # occurs in exactly one head; padded source slots are excluded.
        return relation.transpose(1, 2)[:, :, self.indices] & self.live

    def _local_update(self, state, mask):
        senders = state[:, self.indices]
        # Scalar Q=(1+h_i)*q uses an immutable 1 coordinate. K=h_j*k.
        # The real values are the source bits, and the appended null is zero.
        queries = (1 + state[:, :, None]) * self.q
        keys = senders * self.k[:, None]
        scores = (queries[:, :, :, None] * keys[:, None]).masked_fill(~mask, -torch.inf)
        probabilities = F.pad(scores, (0, 1), value=0).softmax(-1)[..., :-1]
        values = senders * self.v[:, None]
        contexts = (probabilities * values[:, None]).sum(-1)
        return (contexts * self.o).sum(-1)

    def _global_update(self, state, key_coordinates):
        # Each node holds a relation row. Its dot product with vertex j's
        # ID coordinate reads that row's membership bit for j. A null key
        # supplies zero whenever no reached vertex belongs to a head shard.
        queries = F.pad(state, (0, 1), value=1)[:, None] * self.q[None, :, None, None]
        keys = key_coordinates * self.k[:, None, None]
        scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(self.nodes+1)
        keep = F.pad(self.live, (0, 1), value=True)
        probabilities = scores.masked_fill(~keep[None, :, None, :], -torch.inf).softmax(-1)
        values = state[:, self.indices, :] * self.v[None, :, None, None]
        contexts = probabilities @ F.pad(values, (0, 0, 0, 1))
        return (contexts * self.o[None, :, None, None]).sum(1)

    def forward(self, state, prepared):
        update = self._global_update(state, prepared) if self.global_attention else self._local_update(state, prepared)
        residual = state + update
        return residual + self.w2 * torch.relu(self.w1*residual + self.b1) + self.b2


class _Reachability(nn.Module):
    def __init__(self, name, nodes, budget, depth, heads, *, tied=False, global_attention=False):
        super().__init__()
        self.name, self.nodes, self.budget = name, nodes, budget
        self.depth, self.tied, self.global_attention = depth, tied, global_attention
        self.heads_per_stage = heads
        self.blocks = nn.ModuleList(_Stage(nodes, heads, global_attention) for _ in range(1 if tied else depth))
        self.readout = _coefficient([-1., 1.])
        self.bias = _coefficient([0.5, -0.5])

    def metadata(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            "name": self.name, "parameters": total, "nonzero_parameters": total-len(self.blocks),
            "budget": self.budget, "exact_budget": total == self.budget, "steps": self.depth,
            "stored_stages": len(self.blocks), "heads_per_stage": self.heads_per_stage,
            "tied": self.tied, "operator": "relation squaring" if self.global_attention else "one-hop propagation",
            "coordinates_per_vertex": self.nodes if self.global_attention else 1,
            "covered_edge_hops": min(self.nodes-1, 2**self.depth) if self.global_attention else self.depth,
            "head_groups": "Each real key belongs only to its vertex-ID residue shard. Every shard is nonempty.",
            "coefficient_accounting": "Four Q/K/V/O scalars per shard, four residual MLP scalars per stored "
                                      "stage, plus four common-shape classifier scalars. Coefficients are shared "
                                      "across vertices and, for relation squaring, across relation coordinates.",
            "structural_operations": "Given adjacency, vertex-ID coordinates, identity edges, structural masks, "
                                     "null token, residual sum, softmax, ReLU, and endpoint selection.",
        }

    def forward(self, adjacency, source, target):
        if (adjacency.ndim != 3 or adjacency.shape[1:] != (self.nodes, self.nodes)
                or adjacency.dtype != torch.bool or source.shape != adjacency.shape[:1] or target.shape != source.shape):
            raise ValueError("Expected bool adjacency [B,N,N] and source/target ID vectors [B]")
        bs = adjacency.shape[0]
        vertex = torch.arange(self.nodes, device=adjacency.device)
        # All models receive the same graph with the zero-hop identity relation.
        relation = adjacency | (vertex[:, None] == vertex[None, :])
        dtype = self.readout.dtype
        first = self.blocks[0]
        prepared = first.prepare(relation, dtype)
        if self.global_attention:
            state = relation.to(dtype)
        else:
            state = (vertex[None] == source[:, None]).to(dtype)
        for step in range(self.depth):
            block = first if self.tied else self.blocks[step]
            state = block(state, prepared)
        batch = torch.arange(bs, device=adjacency.device)
        flag = state[batch, source, target] if self.global_attention else state[batch, target]
        return flag[:, None] * self.readout + self.bias


def build_models(budget, nodes=64):
    """Return fixed analytic models. FP64 parameters can be cast with .to()."""
    if budget not in BUDGETS:
        raise ValueError(f"Choose one of the registered budgets {BUDGETS}")
    allocation = (budget-4)//4
    if nodes < allocation-1:
        raise ValueError("Each key shard must include at least one real vertex")
    models = {}
    for name, depth, stored_divisor, tied, glob in (
        ("local_tied", nodes-1, 1, True, False),
        ("local_untied_1", 1, 1, False, False),
        ("local_untied_2", 2, 2, False, False),
        ("local_untied_4", 4, 4, False, False),
        ("global_untied_4", 4, 4, False, True),
        ("global_untied_6", 6, 6, False, True),
        ("local_unrolled", nodes-1, 1, False, False),
    ):
        heads = allocation//stored_divisor-1
        model = _Reachability(name, nodes, budget, depth, heads, tied=tied, global_attention=glob).eval()
        expected = 4 + 4*(heads+1)*(1 if tied else depth)
        if sum(p.numel() for p in model.parameters()) != expected or (name != "local_unrolled" and expected != budget):
            raise AssertionError(f"Coefficient count differs from the design for {name}")
        models[name] = model
    return models
