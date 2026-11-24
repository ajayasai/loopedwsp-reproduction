"""Analytic pointer chasing by softmax attention and binary lifting.

The successor table supplies one categorical ID per node, in node-ID order.
The source and requested hop count are inputs. There are no paths or composed
pointers in the input. A block holds a table T of pointer distributions and
a distribution p for the query. One softmax attention evaluates T(T) for all
rows and T(p) for the query simultaneously, against the same previous T.
At stage j, the j-th binary digit of the requested count chooses between T(p)
and p using a pointwise ReLU gate. Initially T=f and p is the input source.
Thus T represents f^(2^j) at entry to stage j, and p accumulates the selected
powers. We decode node IDs by argmax of the final query scores.

All models have input-independent depth. A D-stage model clips the raw hop
count at 2^D-1 on-device before binary encoding. This gives a defined bounded
answer above its capacity and keeps every reachable step for shallow models.
The dataset uses nonnegative k through 32; six stages can represent 0..63.

Each block stores two independent, dense [nodes, width] matrices for Q and K.
Each successive column pair encodes another Fourier harmonic of node ID.
Every column is active: additional harmonics increase the rank of the node
similarity kernel. ID one-hot features, the identity value/output maps and
class readout, bit extraction, and the two-ReLU gate are fixed operations with
no coefficient arrays. No graph operations happen in the host control flow.
"""

import math
from collections import OrderedDict
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


BUDGETS = (3072, 6144)
MODEL_NAMES = ("loop_shared", "fixed_2", "fixed_4", "fixed_6", "untied_full")


class _PointerBlock(nn.Module):
    def __init__(self, nodes, width):
        super().__init__()
        if width < 2 or width % 2 or width // 2 > (nodes - 1) // 2:
            raise ValueError("Width must contain distinct complete sine/cosine pairs below Nyquist")
        # A generic phase offset avoids zero coefficients. It cancels in the
        # query-key dot product, which depends only on the ID difference.
        node = torch.arange(nodes, dtype=torch.float64)[:, None]
        frequency = torch.arange(1, width // 2 + 1, dtype=torch.float64)[None, :]
        angle = 2 * math.pi * node * frequency / nodes + (frequency + 0.5) / (nodes + 1)
        features = torch.stack((angle.cos(), angle.sin()), dim=-1).reshape(nodes, width)
        # Standard scores equal 1024 * sum_r cos(2*pi*r*(i-j)/nodes).
        # The maximum is unique at i=j. Even the narrowest model (two
        # harmonic pairs on 64 nodes) separates its nearest neighbor by >24.
        features = features * math.sqrt(1024 * math.sqrt(width))
        self.query = nn.Parameter(features.clone(), requires_grad=False)
        self.key = nn.Parameter(features.clone(), requires_grad=False)
        self.temperature_divisor = math.sqrt(width)

    def forward(self, table, current, bit):
        queries = torch.cat((table, current), dim=1) @ self.query
        # Prescaling computes standard QK/sqrt(width) without overflowing
        # an intermediate FP16 dot product at the larger widths.
        scores = (queries / self.temperature_divisor) @ self.key.T
        updated = torch.softmax(scores, dim=-1) @ table
        candidate = updated[:, -1:, :]
        # For binary b and a probability vector x, ReLU(x-(1-b)) is b*x.
        # Route the new query for b=1, or the previous query for b=0.
        current = F.relu(candidate - (1 - bit)) + F.relu(current - bit)
        return updated[:, :-1, :], current


class PointerTransformer(nn.Module):
    def __init__(self, name, nodes, width, stages, shared):
        super().__init__()
        self.name = name
        self.nodes = nodes
        self.width = width
        self.stages = stages
        self.shared = shared
        self.blocks = nn.ModuleList(_PointerBlock(nodes, width) for _ in range(1 if shared else stages))

    def executed_depth(self, hops):
        """Number of attention/propagation stages, identical for all requested hop counts."""
        return self.stages

    def metadata(self):
        dense = sum(p.numel() for p in self.parameters())
        nonzero = sum(int(torch.count_nonzero(p).item()) for p in self.parameters())
        return {
            "name": self.name, "nodes": self.nodes, "address_width": self.width,
            "stages": self.stages, "distinct_blocks": len(self.blocks), "shared": self.shared,
            "hop_capacity": (1 << self.stages) - 1, "parameters": dense,
            "nonzero_parameters": nonzero, "embedding_parameters": 0, "core_parameters": dense,
            "readout_parameters": 0, "trainable_parameters": 0,
            "block_parameters": 2 * self.nodes * self.width,
            "coefficients": "Two independent dense node-by-address-width Q and K matrices per distinct block.",
            "fixed_operations": "One-hot IDs; one dot-product softmax over all node keys; identity V/O; two-ReLU binary gate; clipped integer hop-bit encoding; identity node-class readout.",
        }

    def forward(self, successor, source, hops):
        if successor.ndim != 2 or successor.shape[1] != self.nodes or source.shape != successor.shape[:1]:
            raise ValueError("Expected successor [batch,nodes] and source [batch]")
        if successor.dtype != torch.long or source.dtype != torch.long or successor.device != source.device:
            raise ValueError("Successor and source must be same-device int64 tensors")
        if not isinstance(hops, (Integral, torch.Tensor)):
            raise TypeError("Hop count must be an integer or an integer tensor")
        hops = torch.as_tensor(hops, device=successor.device)
        if hops.is_floating_point() or hops.is_complex() or hops.dtype == torch.bool:
            raise TypeError("Hop count must have an integer dtype")
        if hops.ndim != 0 and hops.shape != source.shape:
            raise ValueError("Hop count must be a scalar or have one entry per record")
        capped = hops.to(torch.long).clamp(0, (1 << self.stages) - 1)
        dtype = self.blocks[0].query.dtype
        table = F.one_hot(successor, self.nodes).to(dtype)
        current = F.one_hot(source, self.nodes).to(dtype)[:, None, :]
        for index in range(self.stages):
            bit = ((capped >> index) & 1).to(dtype).reshape(-1, 1, 1)
            block = self.blocks[0] if self.shared else self.blocks[index]
            table, current = block(table, current, bit)
        return current[:, 0, :]


def build_models(budget, nodes=64):
    """Return the four exactly matched designs and a wider untied saturation control.

    At 64 nodes, budgets 3072 and 6144 give wide shared widths 24 and 48.
    A D-layer fixed design splits that width over its D independent blocks.
    The six-wide-block control stores six times the requested budget.
    """
    if budget not in BUDGETS or nodes != 64:
        raise ValueError(f"The registered pointer budgets are {BUDGETS} at 64 nodes")
    width = budget // (2 * nodes)
    models = OrderedDict()
    models["loop_shared"] = PointerTransformer("loop_shared", nodes, width, 6, shared=True)
    for depth in (2, 4, 6):
        name = f"fixed_{depth}"
        models[name] = PointerTransformer(name, nodes, width // depth, depth, shared=False)
    models["untied_full"] = PointerTransformer("untied_full", nodes, width, 6, shared=False)
    return models
