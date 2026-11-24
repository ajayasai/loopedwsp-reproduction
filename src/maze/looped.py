"""Generic inference for a residual softmax-attention/ReLU block with tied weights."""
from __future__ import annotations

import math

import numpy as np


def positions(weights):
    return (weights["row"][:, None, :] + weights["col"][None, :, :]).reshape(
        -1, weights["token"].shape[-1]
    )


def dense_numpy(weights, ids, rounds):
    """Literal float64 implementation for small independent equivalence checks."""
    h = weights["token"][ids] + positions(weights)
    for _ in range(rounds):
        for layer in weights["layers"]:
            update = np.zeros_like(h)
            for q, k, v, o in zip(*(layer[key] for key in ("q", "k", "v", "o"))):
                scores = (h @ q) @ (h @ k).swapaxes(-1, -2) / math.sqrt(q.shape[-1])
                exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
                update += ((exp / exp.sum(axis=-1, keepdims=True)) @ (h @ v)) @ o
            h = h + update
            h = h + np.maximum(0, h @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
    return h @ weights["readout"] + weights["bias"]


class TorchBlock:
    """Prepare invariant attention from weight dependencies, without interpreting tokens."""

    def __init__(self, w, dtype, device):
        import torch

        self.torch = torch
        self.tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        self.token = self.tensor(w["token"])
        self.pos = self.tensor(positions(w))
        self.readout = self.tensor(w["readout"])
        self.bias = self.tensor(w["bias"])
        variable = np.any(w["token"] != w["token"][:1], axis=0)
        for layer in w["layers"]:
            variable |= np.any(layer["o"] != 0, axis=(0, 1))
            variable |= np.any(layer["w2"] != 0, axis=0) | (layer["b2"] != 0)

        reference = self.token[0] + self.pos
        self.layers = []
        self.head_types = []
        for layer in w["layers"]:
            heads = []
            for q, k, v, o in zip(*(layer[key] for key in ("q", "k", "v", "o"))):
                keep = np.any(v != 0, axis=0) & np.any(o != 0, axis=1)
                if not np.any(keep):
                    continue
                scale = math.sqrt(q.shape[1])
                static = not (np.any(q[variable] != 0) or np.any(k[variable] != 0))
                q, k = self.tensor(q), self.tensor(k)
                route = attention = None
                if static:
                    scores = (reference @ q) @ (reference @ k).T / scale
                    attention = scores.softmax(dim=-1)
                    maxima, indices = attention.max(dim=-1)
                    if torch.all(maxima == 1).item() and torch.count_nonzero(attention).item() == len(indices):
                        route, attention = indices, None
                heads.append((q, k, self.tensor(v[:, keep].copy()), self.tensor(o[keep].copy()),
                              scale, route, attention))
                self.head_types.append("exact one-hot" if route is not None else
                                       "constant dense" if static else "dynamic dense")
            ff = [self.tensor(layer[key]) for key in ("w1", "b1", "w2", "b2")]
            projection = torch.cat([head[3] for head in heads], dim=0) if heads else None
            self.layers.append((heads, projection, ff))

    def step(self, h):
        torch = self.torch
        for heads, projection, (w1, b1, w2, b2) in self.layers:
            contributions = []
            for q, k, v, _, scale, route, attention in heads:
                values = h @ v
                if route is not None:
                    values = values.index_select(1, route)
                elif attention is not None:
                    values = attention @ values
                else:
                    scores = (h @ q) @ (h @ k).transpose(-1, -2) / scale
                    values = scores.softmax(dim=-1) @ values
                contributions.append(values)
            if contributions:
                h = h + torch.cat(contributions, dim=-1) @ projection
            h = h + torch.relu(h @ w1 + b1) @ w2 + b2
        return h


def gpu_forward(w, ids, checkpoints, dtype="float64", batch_size=64, diagnostics=None):
    """Evaluate fixed loop budgets; elide repeats only after an exact one-step identity."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the benchmark")
    milestones = sorted(set(checkpoints))
    if not milestones or milestones[0] < 0 or batch_size < 1:
        raise ValueError("invalid checkpoint or batch size")
    ids = np.asarray(ids, dtype=np.int64)
    device = torch.device("cuda")
    results = {t: [] for t in milestones}
    steps = []
    with torch.inference_mode():
        model = TorchBlock(w, getattr(torch, dtype), device)
        for start in range(0, len(ids), batch_size):
            batch = torch.as_tensor(ids[start:start + batch_size], device=device)
            h = model.token[batch] + model.pos
            t = 0
            fixed = False
            for target in milestones:
                while t < target and not fixed:
                    new = model.step(h)
                    t += 1
                    if t % 16 == 0:
                        fixed = torch.equal(new, h)
                    h = new
                decoded = (h @ model.readout + model.bias).argmax(dim=-1)
                results[target].append(decoded.cpu().numpy())
            steps.append({"items": len(batch), "evaluated_blocks": t, "full_state_fixed": fixed})
    if diagnostics is not None:
        diagnostics.update({"heads": model.head_types, "batches": steps})
    return {t: np.concatenate(rows) for t, rows in results.items()}
