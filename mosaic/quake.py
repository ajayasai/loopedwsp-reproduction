"""Locate every grid source and origin tick consistent with a synthetic seismic network."""
from math import ceil, hypot, isqrt

import numpy as np
import torch
from torch import nn


SIDE = 7
MAX_ORIGIN = 15
POINTS = [(x, y) for y in range(SIDE) for x in range(SIDE)]
MAX_SQUARED = 2 * (SIDE - 1)**2
MAX_TRAVEL = ceil(MAX_SQUARED**0.5)


def encode(record):
    """Keep the measured station coordinates and times, using -1 for missing."""
    if not isinstance(record, dict) or set(record) != {"stations"}:
        raise ValueError("Supply a stations array")
    rows = record["stations"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(POINTS):
        raise ValueError("Supply 1–49 stations")
    tokens = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each station is [x, y, first_arrival_tick]")
        x, y, pick = row
        if any(type(v) is not int or not 0 <= v < SIDE for v in (x, y)):
            raise ValueError("Station coordinates must be integers between 0 and 6")
        if pick is not None and (type(pick) is not int or not 0 <= pick <= MAX_ORIGIN + MAX_TRAVEL):
            raise ValueError("An arrival must be null or an integer between 0 and 24")
        tokens.append((x, y, -1 if pick is None else pick))
    if len({(x, y) for x, y, _ in tokens}) != len(tokens):
        raise ValueError("Station positions must be distinct")
    if all(pick == -1 for _, _, pick in tokens):
        raise ValueError("At least one station needs a measured arrival")
    return np.asarray(sorted(tokens), dtype=np.int64)


class ArrivalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Each squared-distance address holds the first sample tick reached.
        times = [isqrt(s) + int(isqrt(s)**2 < s) for s in range(MAX_SQUARED + 1)]
        self.times = nn.Parameter(torch.tensor(times, dtype=torch.float32), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(64.), requires_grad=False)

    def forward(self, tokens):
        stations = tokens.to(self.times.dtype)
        source = torch.tensor(POINTS, dtype=stations.dtype, device=stations.device)
        delta = source[None, :, None, :] - stations[:, None, :, :2]
        squared = delta.square().sum(-1)
        addresses = torch.arange(len(self.times), device=stations.device, dtype=stations.dtype)
        # Dot-product attention for the nearest integer address. The omitted
        # query-only term -squared**2 cancels during softmax normalization.
        query = torch.stack((2*squared, torch.ones_like(squared)), dim=-1)
        keys = torch.stack((addresses, -addresses.square()))
        weights = (self.sharpness * (query @ keys)).softmax(-1)
        origins = stations[:, None, :, 2] - weights @ self.times
        missing = (stations[:, None, :, 2] < 0)

        # Reading both extreme implied origin times preserves ties between
        # candidate source cells. Missing stations never serve as keys.
        scores = self.sharpness * origins
        latest = (scores.masked_fill(missing, -torch.inf).softmax(-1) * origins).sum(-1)
        earliest = ((-scores).masked_fill(missing, -torch.inf).softmax(-1) * origins).sum(-1)
        valid = ((latest-earliest < 0.5) & (earliest > -0.5) & (latest < MAX_ORIGIN+0.5))
        labels = torch.where(valid, earliest+1, torch.zeros_like(earliest))
        classes = torch.arange(MAX_ORIGIN+2, device=stations.device, dtype=stations.dtype)
        return -(labels.unsqueeze(-1) - classes).abs()


def build_model():
    return ArrivalTransformer().eval()


def reference(record):
    """Intersect origin times implied by each measured arrival at each candidate."""
    observed = [row for row in record["stations"] if row[2] is not None]
    labels = []
    for x, y in POINTS:
        origins = {pick - ceil(hypot(x-sx, y-sy)) for sx, sy, pick in observed}
        origin = next(iter(origins)) if len(origins) == 1 else -1
        labels.append(origin+1 if 0 <= origin <= MAX_ORIGIN else 0)
    return np.asarray(labels, dtype=np.int64)


def generate(rng, size, index):
    source_id = int(rng.integers(len(POINTS)))
    x, y = POINTS[source_id]
    origin = int(rng.integers(MAX_ORIGIN+1))
    delays = np.array([ceil(hypot(x-sx, y-sy)) for sx, sy in POINTS])
    mode = index % 5
    if mode == 2:
        # Favor a station layout that cannot separate two adjacent sources.
        neighbors = [(a,b) for a,b in POINTS if abs(a-x)+abs(b-y) == 1]
        ax, ay = neighbors[int(rng.integers(len(neighbors)))]
        alt = [ceil(hypot(ax-sx, ay-sy)) for sx, sy in POINTS]
        common = np.flatnonzero(delays == alt)
        selected = rng.choice(common, min(size, len(common)), replace=False).tolist()
        others = np.setdiff1d(np.arange(len(POINTS)), selected)
        missing = set(rng.choice(others, size-len(selected), replace=False).tolist())
        selected.extend(sorted(missing))
    else:
        selected = rng.choice(len(POINTS), size, replace=False).tolist()
        missing = {selected[-1]} if mode == 1 else set()
    rows = [[*POINTS[i], None if i in missing else int(origin + delays[i])] for i in selected]
    if mode == 3:
        j = int(rng.integers(size))
        # A one-tick pick error can produce a tie, move the answer or eliminate it.
        rows[j][2] += 1 if rows[j][2] < MAX_ORIGIN+MAX_TRAVEL else -1
    if mode == 4:
        rows[0][2], rows[1][2] = 1, 22
    return {"stations": sorted(rows)}


def example():
    return {"stations": [[0,0,8], [6,0,9], [0,6,8]]}


def format_output(record, prediction):
    if len(prediction) != len(POINTS) or any(not 0 <= value <= MAX_ORIGIN+1 for value in prediction):
        raise ValueError("Expected one valid origin class per candidate source")
    solutions = [{"x":x, "y":y, "origin_tick":int(value)-1}
                 for (x,y),value in zip(POINTS,prediction,strict=True) if value]
    return {"solutions": solutions}
