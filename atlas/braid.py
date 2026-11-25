"""Count Fox 3-colorings of the closure of a signed three-strand braid.

A recurrent block transports all 27 possible top color triples through the
word. Crossing tokens query four fixed linear maps with softmax attention.
A ReLU spline reduces their signed outputs modulo three at each step. The
readout counts triples whose bottom colors agree with their own top colors.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MAX_CROSSINGS = 64
CLASSES = 28
GENERATORS = (-2, -1, 1, 2)


def encode(record):
    if not isinstance(record, dict) or set(record) != {"word"}:
        raise ValueError("Supply a word on the three-strand braid generators")
    word = record["word"]
    if (not isinstance(word, list) or len(word) > MAX_CROSSINGS
            or any(type(c) is not int or c not in GENERATORS for c in word)):
        raise ValueError("Use at most 64 signed crossings from -2, -1, 1, 2")
    return np.asarray(word, dtype=np.int64).reshape(-1, 1)


class BraidColorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Values act on color columns. Positive crossing on (a,b) leaves
        # (2a-b,a) in the two bottom positions; the negative is its inverse.
        values = []
        for generator in GENERATORS:
            matrix = torch.eye(3)
            i = abs(generator) - 1
            matrix[i:i + 2, i:i + 2] = torch.tensor(
                [[2., -1.], [1., 0.]] if generator > 0 else [[0., 1.], [-1., 2.]]
            )
            values.append(matrix)
        self.symbols = nn.Parameter(torch.tensor(GENERATORS, dtype=torch.float32), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(128.), requires_grad=False)
        self.crossing_values = nn.Parameter(torch.stack(values), requires_grad=False)
        # On the integers -2 through 4 the spline is exactly the residue in {0,1,2}.
        self.wrap_offsets = nn.Parameter(torch.tensor([1., 0., -2., -3.]), requires_grad=False)
        self.wrap_weights = nn.Parameter(torch.tensor([-3., 3., -3., 3.]), requires_grad=False)
        self.wrap_bias = nn.Parameter(torch.tensor(3.), requires_grad=False)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 1 or tokens.shape[1] > MAX_CROSSINGS:
            raise ValueError("Expected [batch, crossings, 1] signed braid tokens")
        raw = tokens.to(self.symbols.dtype)
        queries = torch.cat((2 * raw, torch.ones_like(raw)), dim=-1)
        keys = torch.stack((self.symbols, -self.symbols.square()))
        attention = ((queries @ keys) * self.sharpness).softmax(-1)
        maps = (attention @ self.crossing_values.flatten(1)).unflatten(-1, (3, 3))

        digits = torch.arange(3, device=raw.device, dtype=raw.dtype)
        starts = torch.cartesian_prod(digits, digits, digits).expand(raw.shape[0], -1, -1)
        colors = starts
        for crossing in maps.unbind(1):
            linear = colors @ crossing.transpose(-1, -2)
            wrap = F.relu(linear.unsqueeze(-1) + self.wrap_offsets) @ self.wrap_weights
            colors = linear + wrap + self.wrap_bias
        closed = F.relu(1 - (colors - starts).abs().sum(-1))
        count = closed.sum(-1, keepdim=True)
        classes = torch.arange(CLASSES, device=raw.device, dtype=raw.dtype)
        return -(count.unsqueeze(-1) - classes).abs()


def build_model():
    return BraidColorTransformer().eval()


def reference(record):
    """Count the kernel of (transport - identity) by row reduction over GF(3)."""
    encode(record)
    transport = np.eye(3, dtype=np.int64)
    for crossing in record["word"]:
        i = abs(crossing) - 1
        a, b = transport[i:i + 2].copy()
        if crossing > 0:
            transport[i:i + 2] = [(2 * a - b) % 3, a]
        else:
            transport[i:i + 2] = [b, (2 * b - a) % 3]

    residual = (transport - np.eye(3, dtype=np.int64)) % 3
    rank = 0
    for column in range(3):
        pivot = next((r for r in range(rank, 3) if residual[r, column]), None)
        if pivot is None:
            continue
        residual[[rank, pivot]] = residual[[pivot, rank]]
        # Both nonzero elements of GF(3) are their own multiplicative inverses.
        residual[rank] = (residual[rank] * residual[rank, column]) % 3
        for r in range(rank + 1, 3):
            residual[r] = (residual[r] - residual[r, column] * residual[rank]) % 3
        rank += 1
    return np.array([3 ** (3 - rank)], dtype=np.int64)


def generate(rng, size, index):
    """Mix random words, cancellation controls and stabilized trefoil words."""
    if not 4 <= size <= MAX_CROSSINGS or size % 2:
        raise ValueError("The study generator uses even lengths from four through 64")
    if index % 4 < 2:
        word = rng.choice(GENERATORS, size=size).tolist()
    else:
        # Insert an inverse pair at a random position without changing the
        # braid. Cyclic shifts preserve the isotopy type of its closure.
        word = []
        if index % 4 == 3:
            a = int(rng.choice(GENERATORS))
            b = int(rng.choice((-1, 1))) * (3 - abs(a))
            word = [a, a, a, b]
        while len(word) < size:
            a = int(rng.choice(GENERATORS))
            start = int(rng.integers(len(word) + 1))
            word[start:start] = [a, -a]
        offset = int(rng.integers(size))
        word = word[offset:] + word[:offset]
    return {"word": word}


def example():
    return {"word": [1, 1, 1, 2]}


def format_output(record, prediction):
    encode(record)
    if len(prediction) != 1 or int(prediction[0]) not in (3, 9, 27):
        raise ValueError("Expected one three-strand coloring count")
    count = int(prediction[0])
    return {"fox_3_colorings": count, "nonconstant_colorings": count - 3}
