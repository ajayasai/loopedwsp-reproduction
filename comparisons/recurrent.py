"""Dense, native-dtype inference with the original frozen application tensors."""

import math

import torch
from torch import nn


def _frozen(array):
    return nn.Parameter(torch.as_tensor(array).detach().clone(), requires_grad=False)


class _Head(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.q = _frozen(weights[prefix + "q"])
        self.k = _frozen(weights[prefix + "k"])
        self.v = _frozen(weights[prefix + "v"])
        self.o = _frozen(weights[prefix + "o"])
        self.scale = math.sqrt(self.q.shape[1])

    def forward(self, state):
        query, key, value = state @ self.q, state @ self.k, state @ self.v
        # Apply the divisor after the score matmul, as in the frozen FP64 frame.
        # A negative overflow could otherwise disappear inside a finite softmax.
        scores = (query @ key.transpose(-1, -2)) / self.scale
        attention = scores.softmax(dim=-1)
        context = attention @ value
        update = context @ self.o
        valid = torch.isfinite(scores).all(dim=-1) & torch.isfinite(update).all(dim=-1)
        return update, valid


class _Layer(nn.Module):
    def __init__(self, weights, prefix, heads):
        super().__init__()
        self.heads = nn.ModuleList(
            _Head(weights, prefix + f"h{head}.") for head in range(heads)
        )
        self.w1 = _frozen(weights[prefix + "w1"])
        self.b1 = _frozen(weights[prefix + "b1"])
        self.w2 = _frozen(weights[prefix + "w2"])
        self.b2 = _frozen(weights[prefix + "b2"])

    def forward(self, state, valid):
        residual = state
        for head in self.heads:
            # Every head sees the same layer input.
            update, head_valid = head(state)
            valid = valid & head_valid
            residual = residual + update
        linear = residual @ self.w1 + self.b1
        state = residual + torch.relu(linear) @ self.w2 + self.b2
        # Check before ReLU: a negative infinity can become a finite zero.
        valid = valid & torch.isfinite(linear).all(dim=-1) & torch.isfinite(state).all(dim=-1)
        return state, valid


class _FrozenRecurrent(nn.Module):
    def __init__(self, spec, weights):
        super().__init__()
        self.embeddings = nn.ParameterList(
            _frozen(weights[f"e{i}"]) for i in range(len(spec["fields"]))
        )
        self.layers = nn.ModuleList(
            _Layer(weights, f"l{i}.", heads) for i, heads in enumerate(spec["layers"])
        )
        self.readout = _frozen(weights["readout"])
        self.bias = _frozen(weights["bias"])
        if self.bias.shape != (2,) or self.readout.shape[-1] != 2:
            raise ValueError("Expected two per-token classification logits")
        if sum(p.numel() for p in self.parameters()) != sum(w.size for w in weights.values()):
            raise ValueError("The module did not consume every frozen tensor entry")

    def forward(self, x, steps, return_raw=False):
        """Return logit[1] - logit[0]; NaN marks an invalid token prediction."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if x.ndim != 3 or x.shape[2] != len(self.embeddings) or x.dtype != torch.int64:
            raise ValueError("Expected an int64 batch of shape [batch, tokens, fields]")
        state = self.embeddings[0][x[:, :, 0]]
        for i in range(1, len(self.embeddings)):
            state = state + self.embeddings[i][x[:, :, i]]
        valid = torch.isfinite(state).all(dim=-1)
        # Each call starts from raw embeddings. No cross-call warm state,
        # fixed-point skipping, gathered attention, autocast, or dtype changes.
        for _ in range(steps):
            for layer in self.layers:
                state, valid = layer(state, valid)
        logits = state @ self.readout + self.bias
        scores = logits[..., 1] - logits[..., 0]
        raw_valid = torch.isfinite(logits).all(dim=-1) & torch.isfinite(scores)
        raw = scores.masked_fill(~raw_valid, float("nan"))
        screened = raw.masked_fill(~valid, float("nan"))
        return (screened, raw) if return_raw else screened


def build_recurrent(spec, weights):
    """Build once from the archive. Use .to(device=..., dtype=...) before inference."""
    return _FrozenRecurrent(spec, weights).eval()
