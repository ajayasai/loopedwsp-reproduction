"""All minimum-change ancestral plumage states on a supplied, synthetic binary tree."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STATES = ("plain", "barred", "spotted")
MAX_TIPS = 32


def encode(record):
    if not isinstance(record, dict) or set(record) != {"tips", "joins"}:
        raise ValueError("Supply observed tips and ordered pairs of children in joins")
    tips, joins = record["tips"], record["joins"]
    if not isinstance(tips, list) or not 2 <= len(tips) <= MAX_TIPS or any(s not in STATES for s in tips):
        raise ValueError("Supply 2 to 32 tips, each plain, barred or spotted")
    if not isinstance(joins, list) or len(joins) != len(tips) - 1:
        raise ValueError("Supply one fewer join than tips")
    tokens = [[STATES.index(s), -1, -1] for s in tips]
    children = []
    for node, pair in enumerate(joins, start=len(tips)):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(child) is not int or not 0 <= child < node for child in pair)):
            raise ValueError("Each join needs two earlier node indices")
        children.extend(pair)
        tokens.append([-1, *pair])
    if sorted(children) != list(range(len(tokens) - 1)):
        raise ValueError("Every node except the last root must have exactly one parent")
    return np.asarray(tokens, dtype=np.int64)


class PlumageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.address_scale = nn.Parameter(torch.tensor(128.), requires_grad=False)
        self.change_cost = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.state_bits = nn.Parameter(torch.tensor([1., 2., 4.]), requires_grad=False)

    @staticmethod
    def lowest(cost):
        # A two-layer ReLU minimum across the three possible patterns.
        a, b, c = cost.split(1, dim=-1)
        ab = a - F.relu(a - b)
        return ab - F.relu(ab - c)

    def edge_message(self, cost):
        # Given the state at the adjacent node, either keep it or pay for a change.
        ceiling = self.lowest(cost) + self.change_cost
        return cost - F.relu(cost - ceiling)

    def forward(self, tokens):
        nodes = tokens.shape[1]
        tips = (nodes + 1) // 2
        raw = tokens.to(self.change_cost.dtype)
        addresses = torch.arange(nodes, device=raw.device, dtype=raw.dtype)
        keys = torch.stack((addresses, -addresses.square()))
        child_ids = raw[:, :, 1:]
        queries = torch.stack((2*child_ids, torch.ones_like(child_ids)), dim=-1)
        # Mask after softmax so that a leaf, which has no children, reads zero.
        edges = ((queries @ keys) * self.address_scale).softmax(-1)
        edges = edges * (raw[:, :, 0] < 0)[:, :, None, None]
        left, right = edges.unbind(2)
        children = left + right
        observed = raw[:, :, :1]
        states = torch.arange(3, device=raw.device, dtype=raw.dtype)
        tip_cost = ((observed >= 0) & (observed != states)).to(raw.dtype) * (nodes + 1)

        # Inside costs count the minimum number of changes below each node,
        # conditional on each possible state at that node.
        inside = tip_cost
        for _ in range(tips - 1):
            inside = tip_cost + children @ self.edge_message(inside)

        # A child's outside cost includes the parent side and the *other* child.
        # Transposing an address routes information back along the same tree edge.
        child_cost = self.edge_message(inside)
        sibling = left.transpose(1, 2) @ (right @ child_cost) + right.transpose(1, 2) @ (left @ child_cost)
        parent = children.transpose(1, 2)
        outside = torch.zeros_like(inside)
        for _ in range(tips - 1):
            outside = self.edge_message(parent @ outside + sibling)

        total = (inside + outside)[:, tips:, :]
        selected = F.relu(self.change_cost - (total - self.lowest(total)))
        mask = selected @ self.state_bits
        classes = torch.arange(8, device=raw.device, dtype=raw.dtype)
        return -(mask.unsqueeze(-1) - classes).abs()


def build_model():
    return PlumageTransformer().eval()


def reference(record):
    """Scalar subtree-cost recursion with traceback through every tied parent choice."""
    encode(record)
    tips, joins = record["tips"], record["joins"]
    limit = 2 * len(tips)
    costs = [[0 if s == observed else limit for s in STATES] for observed in tips]
    for left, right in joins:
        parent_cost = []
        for parent_state in range(3):
            parent_cost.append(sum(min(value + (state != parent_state)
                                       for state, value in enumerate(costs[child])) for child in (left, right)))
        costs.append(parent_cost)

    possible = [set() for _ in costs]
    possible[-1] = {s for s in range(3) if costs[-1][s] == min(costs[-1])}
    for offset in reversed(range(len(joins))):
        for child in joins[offset]:
            for parent_state in possible[len(tips) + offset]:
                conditional = [value + (s != parent_state) for s, value in enumerate(costs[child])]
                possible[child].update(s for s in range(3) if conditional[s] == min(conditional))
    return np.asarray([sum(1 << s for s in choices) for choices in possible[len(tips):]], dtype=np.int64)


def generate(rng, size, index):
    # Mixed topologies include deep caterpillars, near-balanced trees and
    # uniformly chosen pairs of active nodes. Tip IDs and child order vary.
    active = rng.permutation(size).tolist()
    joins = []
    for node in range(size, 2*size - 1):
        if index % 3 == 0:
            pair = [active.pop(), active.pop()]
            active.append(node)
        elif index % 3 == 1:
            pair = [active.pop(0), active.pop(0)]
            active.append(node)
        else:
            pair = [active.pop(int(rng.integers(len(active)))) for _ in range(2)]
            active.append(node)
        if rng.integers(2):
            pair.reverse()
        joins.append(pair)
    if index % 4 == 0:
        # Simulate some shared patterns by occasionally changing state down a branch.
        colors = [0] * (2*size - 1)
        colors[-1] = int(rng.integers(3))
        for k in reversed(range(len(joins))):
            for child in joins[k]:
                colors[child] = (colors[size+k] + int(rng.integers(1, 3))) % 3 if rng.random() < .25 else colors[size+k]
    else:
        colors = rng.integers(3, size=size).tolist()
    return {"tips":[STATES[c] for c in colors[:size]], "joins":joins}


def example():
    return {"tips":["plain","barred","plain","plain"], "joins":[[0,1],[2,3],[4,5]]}


def format_output(record, prediction):
    encode(record)
    if len(prediction) != len(record["joins"]) or any(not 1 <= mask <= 7 for mask in prediction):
        raise ValueError("Expected one nonempty state set per internal node")
    return {"ancestors":[{"node": node, "states":[state for bit,state in enumerate(STATES) if mask & (1 << bit)]}
                         for node,mask in enumerate(prediction, start=len(record["tips"]))]}
