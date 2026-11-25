"""Fill closed surface depressions on a small integer elevation raster.

The perimeter drains freely and water travels between horizontally or
vertically adjacent cells. A cell's spill elevation is the smallest possible
maximum ground elevation along a path from that cell to the perimeter.
Depth is the difference between spill elevation and ground elevation. All
cells have unit area, so summing depth gives volume in cell-height units.

The tied attention head reads the lowest current spill level in its
five-cell window (four neighbors plus the cell itself). A residual ReLU
enforces the ground as a lower bound. Interior cells start at the highest
allowed elevation and the perimeter starts at its ground level. Each pass
extends paths to a drain by one edge. A shortest minimax path never needs
to repeat an interior cell, so (side-2)^2 passes suffice. Ground, grid
geometry and integer class coordinates are fixed inputs or operations.
The reference uses a separate heap-based priority flood.
"""
import heapq
from math import isqrt

import numpy as np


def encode(record):
    if not isinstance(record, dict):
        raise ValueError("Expected an object containing heights")
    heights = np.asarray(record.get("heights"))
    if (heights.ndim != 2 or heights.shape[0] != heights.shape[1]
            or not 3 <= heights.shape[0] <= 12 or heights.dtype.kind not in "iu"
            or heights.min() < 0 or heights.max() > 9):
        raise ValueError("heights must be a square 3..12 cells wide with integer elevations 0..9")
    return heights.astype(np.int64).reshape(-1, 1)


def build_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SpillBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Parameter(torch.tensor(-128.), requires_grad=False)
            self.value = nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.residual = nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)

        def forward(self, ground, spill, neighbors):
            levels = spill[:, neighbors]
            # A constant query attends to the lowest spill among the local keys.
            # With this FP32 scale, a unit-separated higher key gets zero weight.
            weights = (self.query * levels).softmax(dim=-1)
            gathered = (weights * (self.value * levels)).sum(dim=-1)
            return ground + F.relu(self.residual[0] * gathered + self.residual[1] * ground)

    class BasinTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = SpillBlock()
            self.readout = nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)

        def forward(self, tokens):
            if tokens.ndim != 3 or tokens.shape[-1] != 1:
                raise ValueError("Expected one height feature per raster cell")
            n = isqrt(tokens.shape[1])
            if n*n != tokens.shape[1] or not 3 <= n <= 12:
                raise ValueError("Expected a supported square grid")
            ground = tokens[..., 0].to(self.readout.dtype)
            cell = torch.arange(n*n, device=tokens.device)
            y, x = cell // n, cell % n
            # At a border, a step outside the grid duplicates the border cell.
            neighbors = torch.stack((cell, cell - (y > 0)*n, cell + (y < n-1)*n,
                                     cell - (x > 0).long(), cell + (x < n-1).long()), dim=-1)
            border = (y == 0) | (x == 0) | (y == n-1) | (x == n-1)
            spill = torch.where(border, ground, 9.)
            for _ in range((n-2)**2):
                spill = self.block(ground, spill, neighbors)
            depth = self.readout[0] * spill + self.readout[1] * ground
            ids = torch.arange(10, device=tokens.device, dtype=depth.dtype)
            return -(depth.unsqueeze(-1) - ids).abs()

    return BasinTransformer().eval()


def reference(record):
    flat = encode(record)[:, 0]
    n = isqrt(len(flat))
    grid = flat.reshape(n, n)
    seen = np.zeros((n, n), dtype=bool)
    depths = np.zeros_like(grid)
    queue = []
    for y, x in np.ndindex(n, n):
        if y in (0, n-1) or x in (0, n-1):
            seen[y, x] = True
            heapq.heappush(queue, (int(grid[y, x]), y, x))
    while queue:
        level, y, x = heapq.heappop(queue)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = y + dy, x + dx
            if 0 <= row < n and 0 <= col < n and not seen[row, col]:
                seen[row, col] = True
                elevated = max(level, int(grid[row, col]))
                depths[row, col] = elevated - grid[row, col]
                heapq.heappush(queue, (elevated, row, col))
    return depths.ravel()


def example():
    # The single height-3 saddle in the rim sets the spill level of the interior.
    return {"heights": [[9, 9, 3, 9, 9], [9, 0, 2, 0, 9], [9, 0, 8, 0, 9],
                        [9, 0, 0, 0, 9], [9, 9, 9, 9, 9]]}


def generate(rng, size, index):
    if not 3 <= size <= 12:
        raise ValueError("Unsupported raster size")
    family = index % 4
    if family == 0:
        heights = rng.integers(0, 10, size=(size, size))
    else:
        heights = rng.integers(7, 10, size=(size, size))
        saddle = int(rng.integers(1, 7))
        if family == 2:
            # A single rim outlet and a winding interior channel with high walls.
            heights[1, 0] = saddle
            for turn, row in enumerate(range(1, size-1, 2)):
                heights[row, 1:-1] = rng.integers(0, 3, size=size-2)
                if row + 2 < size-1:
                    col = size-2 if turn % 2 == 0 else 1
                    heights[row+1, col] = int(rng.integers(0, 3))
        else:
            heights[1:-1, 1:-1] = rng.integers(0, 6, size=(size-2, size-2))
            heights[int(rng.integers(1, size-1)), 0] = saddle
            if family == 3 and size >= 5:
                # Two basins separated by a ridge with one interior saddle.
                heights[1:-1, size//2] = rng.integers(7, 10, size=size-2)
                heights[int(rng.integers(1, size-1)), size//2] = int(rng.integers(4, 8))
    heights = np.rot90(heights, int(rng.integers(4)))
    if rng.integers(2):
        heights = np.fliplr(heights)
    return {"heights": heights.tolist()}


def format_output(record, prediction):
    n = len(record["heights"])
    depth = np.asarray(prediction, dtype=np.int64).reshape(n, n)
    return {"depth": depth.tolist(), "volume": int(depth.sum())}
