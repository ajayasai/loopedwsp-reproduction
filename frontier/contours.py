"""Trace cell-edge connections of a bilinearly interpolated elevation contour."""

from fractions import Fraction

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LABELS = ("empty", "N-E", "E-S", "S-W", "W-N", "N-S", "E-W",
          "N-E & S-W", "W-N & E-S", "four-way saddle")
MAX_SIDE = 65
MAX_DIFFERENCE = 8


def encode(record):
    """Store the level in a header row followed by the original vertex samples."""
    if not isinstance(record, dict) or set(record) != {"level", "heights"}:
        raise ValueError("Supply a contour level and a rectangular grid of heights.")
    level, grid = record["level"], record["heights"]
    if type(level) is not int or abs(level) > 4096:
        raise ValueError("The level must be an integer between -4096 and 4096.")
    if (not isinstance(grid, list) or not 2 <= len(grid) <= MAX_SIDE
            or not isinstance(grid[0], list) or not 2 <= len(grid[0]) <= MAX_SIDE):
        raise ValueError("Supply 2 through 65 vertices on each axis.")
    for row in grid:
        if not isinstance(row, list) or len(row) != len(grid[0]):
            raise ValueError("Rows must have equal lengths.")
        if any(type(h) is not int or not 1 <= abs(h-level) <= MAX_DIFFERENCE for h in row):
            raise ValueError("Each integer height must differ from the level by 1 through 8.")
    return np.asarray([[level] * len(grid[0]), *grid], dtype=np.int64)


class ContourTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.address = nn.Parameter(torch.tensor([256., -128.]), requires_grad=False)
        self.bits = nn.Parameter(torch.tensor([1., 2., 8., 4.]), requires_grad=False)
        # Address bits are clockwise NW=1, NE=2, SE=4, SW=8. Value 9 marks
        # the two alternating-sign patterns until the saddle decision is made.
        self.cases = nn.Parameter(torch.tensor(
            [0.,4.,1.,6.,2.,9.,5.,3.,3.,5.,9.,2.,6.,1.,4.,0.]), requires_grad=False)
        # Exact integer square on [-16,16] via the linear spline on |x|.
        self.knots = nn.Parameter(torch.arange(2*MAX_DIFFERENCE, dtype=torch.float32), requires_grad=False)
        self.slopes = nn.Parameter(torch.tensor([1.] + [2.]*(2*MAX_DIFFERENCE-1)), requires_grad=False)

    def addresses(self, length):
        index = torch.arange(length, device=self.address.device, dtype=self.address.dtype)
        targets = torch.stack((index[:-1], index[1:]), dim=-1).flatten()
        q = torch.stack((targets, torch.ones_like(targets)), dim=-1) * self.address
        k = torch.stack((index, index.square()), dim=-1)
        return (q @ k.T).softmax(dim=-1)

    def forward(self, tokens):
        grid = (tokens[:, 1:] - tokens[:, :1]).to(self.address.dtype)
        batch, height, width = grid.shape
        # Axial attention reads the two adjacent rows, then the two adjacent
        # columns, for each cell. Features below are ordered NW, NE, SW, SE.
        rows = self.addresses(height) @ grid
        cells = (rows @ self.addresses(width).T).reshape(batch,height-1,2,width-1,2)
        corners = cells.transpose(2,3).reshape(batch,-1,4)
        code = corners.clamp(0,1) @ self.bits
        ids = torch.arange(16, device=grid.device, dtype=grid.dtype)
        mask = F.relu(1 - (code.unsqueeze(-1)-ids).abs())
        base = mask @ self.cases

        nw, ne, sw, se = corners.unbind(-1)
        sums = torch.stack((nw+se, nw-se, ne+sw, ne-sw), dim=-1)
        squares = F.relu(sums.abs().unsqueeze(-1)-self.knots) @ self.slopes
        aa, ab, ba, bb = squares.unbind(-1)
        determinant = (aa-ab-ba+bb) * .25
        ambiguous = F.relu(base-8)
        positive = F.relu(ambiguous + determinant.clamp(0,1) - 1)
        negative = F.relu(ambiguous + (-determinant).clamp(0,1) - 1)
        label = base - 2*positive - negative
        centers = torch.arange(len(LABELS), device=grid.device, dtype=grid.dtype)
        return -(label.unsqueeze(-1)-centers).abs()


def build_model():
    return ContourTransformer().eval()


def reference(record):
    """Trace sign-changing edges; resolve four crossings via their exact positions."""
    heights, level = record["heights"], record["level"]
    labels = []
    routes = {frozenset(pair): i for i,pair in enumerate(("NE","ES","SW","WN","NS","EW"), 1)}
    for top,bottom in zip(heights,heights[1:]):
        for i in range(len(top)-1):
            a,b,c,d = (z-level for z in (top[i],top[i+1],bottom[i+1],bottom[i]))
            crossings = frozenset(e for e,u,v in zip("NESW",(a,b,c,d),(b,c,d,a)) if (u>0) != (v>0))
            if len(crossings) < 4:
                labels.append(routes[crossings] if crossings else 0)
                continue
            # Both coordinates measure distance from the west edge.
            north, south = Fraction(a,a-b), Fraction(d,d-c)
            labels.append(7 if north > south else 8 if north < south else 9)
    return np.asarray(labels, dtype=np.int64)


def example():
    return {"level": 100, "heights": [[104,99,101,98],[99,101,99,101]]}


def format_output(record, prediction):
    rows, cols = len(record["heights"])-1, len(record["heights"][0])-1
    if len(prediction) != rows*cols or any(not 0 <= v < len(LABELS) for v in prediction):
        raise ValueError("Expected one contour label per cell.")
    return {"cells": [[LABELS[v] for v in prediction[i:i+cols]] for i in range(0, len(prediction), cols)],
            "four_way_saddles": [[r,c] for r in range(rows) for c in range(cols) if prediction[r*cols+c] == 9]}


def generate(rng, size, index):
    """Draw synthetic slopes, hills, irregular fields and planted alternating cells."""
    side = size+1
    y,x = np.indices((side,side))
    cx,cy = rng.integers(side, size=2)
    magnitude = rng.integers(1, MAX_DIFFERENCE+1, size=(side,side))
    mode = index % 4
    if mode == 0:
        sx,sy = rng.choice([-3,-2,-1,1,2,3], size=2)
        raw = sx*(x-cx) + sy*(y-cy)
        field = np.clip(raw + (raw >= 0), -MAX_DIFFERENCE, MAX_DIFFERENCE)
    elif mode == 1:
        radius = int(rng.integers(1,max(2,size//2+1)))
        raw = radius**2 - (x-cx)**2 - (y-cy)**2
        field = np.clip(raw + (raw >= 0), -MAX_DIFFERENCE, MAX_DIFFERENCE)
    elif mode == 2:
        field = magnitude * rng.choice([-1,1], size=(side,side))
    else:
        field = magnitude * (1 - 2*((x+y) % 2))
        a,b = rng.integers(1,MAX_DIFFERENCE+1,size=2)
        r,c = rng.integers(size,size=2)
        field[r:r+2,c:c+2] = [[a,-b],[-a,b]]
    level = int(rng.integers(-100,101))
    if rng.integers(2):
        field = -field
    return {"level":level, "heights":(field+level).tolist()}
