"""Align an eight-ring sample by agreement in the direction of annual changes.

Classes per candidate start are 0 (insufficient), 1 (eligible) and 2
(eligible with the best score). All offsets tied for first receive class 2.
"""
from fractions import Fraction

import numpy as np


SAMPLE_LENGTH = 8
MAX_REFERENCE = 64


def encode(record):
    """Emit the sample followed by the reference as integer width tokens."""
    if not isinstance(record, dict) or set(record) != {"sample", "reference"}:
        raise ValueError("Supply a sample and a reference.")
    sample, master = record["sample"], record["reference"]
    if not isinstance(sample, list) or len(sample) != SAMPLE_LENGTH:
        raise ValueError("The sample must have eight annual positions.")
    if not isinstance(master, list) or not SAMPLE_LENGTH <= len(master) <= MAX_REFERENCE:
        raise ValueError("The reference must have 8 through 64 annual positions.")
    widths = sample + master
    if any(w is not None and (type(w) is not int or not 1 <= w <= 255) for w in widths):
        raise ValueError("Each width must be an integer from 1 to 255 or null.")
    return np.array([[0 if w is None else w] for w in widths], dtype=np.int64)


def build_model():
    import torch
    from torch import nn

    class TreeRingTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.address = nn.Parameter(torch.tensor([256., -128.]), requires_grad=False)
            # 420/n, for the four admissible counts of paired changes.
            self.scale = nn.Parameter(torch.tensor([0., 0., 0., 0., 105., 84., 70., 60.]), requires_grad=False)
            self.rank_gain = nn.Parameter(torch.tensor(128.), requires_grad=False)

        @staticmethod
        def directions(widths):
            left, right = widths[..., :-1], widths[..., 1:]
            observed = torch.minimum(left, right).clamp(0, 1)
            return (right - left).clamp(-1, 1) * observed

        def forward(self, tokens):
            sample = tokens[:, :SAMPLE_LENGTH, 0].to(self.address.dtype)
            master = tokens[:, SAMPLE_LENGTH:].to(self.address.dtype)
            batch, length, _ = master.shape
            places = torch.arange(length, device=master.device, dtype=master.dtype)
            starts = places[:length - SAMPLE_LENGTH + 1]
            desired = (starts[:, None] + places[None, :SAMPLE_LENGTH]).reshape(-1)
            queries = torch.stack((desired, torch.ones_like(desired)), dim=-1) * self.address
            keys = torch.stack((places, places.square()), dim=-1)
            # After subtracting a query constant, the score is -128*(position-desired)^2.
            attention = (queries @ keys.T).softmax(dim=-1)
            aligned = (attention @ master).reshape(batch, len(starts), SAMPLE_LENGTH)
            paired = self.directions(aligned) * self.directions(sample)[:, None, :]
            support = paired.abs().sum(-1).long()
            eligible = (support >= 4).to(master.dtype)
            score = paired.clamp_min(0).sum(-1) * self.scale[support] - (1 - eligible)
            # A second attention read broadcasts the maximum to all offsets. Scores
            # are integers and the softmax margin between distinct values is >=128.
            peak = ((self.rank_gain * score).softmax(-1) * score).sum(-1, keepdim=True)
            best = eligible * (1 - (peak - score).abs().clamp_max(1))
            return torch.stack((1 - eligible, eligible - best, best), dim=-1)

    return TreeRingTransformer().eval()


def reference(record):
    """Use exact rational scores for all eligible sliding windows."""
    encode(record)
    sample, master = record["sample"], record["reference"]
    scores = []
    for start in range(len(master) - len(sample) + 1):
        agree = used = 0
        for i in range(len(sample) - 1):
            a, b = sample[i:i+2]
            c, d = master[start+i:start+i+2]
            if None in (a, b, c, d) or a == b or c == d:
                continue
            used += 1
            agree += (b > a) == (d > c)
        scores.append(Fraction(agree, used) if used >= 4 else None)
    best = max((s for s in scores if s is not None), default=None)
    return np.array([0 if s is None else 2 if s == best else 1 for s in scores], dtype=np.int64)


def format_output(record, prediction):
    if len(prediction) != len(record["reference"]) - SAMPLE_LENGTH + 1:
        raise ValueError("Expected one label for each candidate start.")
    starts = [i for i, value in enumerate(prediction) if value == 2]
    insufficient = [i for i, value in enumerate(prediction) if value == 0]
    status = "insufficient" if not starts else "unique" if len(starts) == 1 else "tied"
    return {"status": status, "best_offsets": starts, "insufficient_offsets": insufficient}


def example():
    return {"reference": [60, 70, 65, 50, 58, 55, 62, 66, 61, 63, 50, 43],
            "sample": [133, 103, 119, 113, 127, 135, 125, 129]}


def generate(rng, size, index):
    """Make artificial sequences with planted windows, repetition, gaps or independent widths."""
    master = rng.integers(4, 95, size=size).tolist()
    start = int(rng.integers(0, size - SAMPLE_LENGTH + 1))
    mode = index % 7
    if mode == 2:
        cycle = rng.choice(np.arange(4, 95), size=int(rng.integers(2, 5)), replace=False).tolist()
        master = [cycle[i % len(cycle)] for i in range(size)]
    scale, shift = int(rng.integers(1, 3)), int(rng.integers(1, 30))
    sample = [w * scale + shift for w in master[start:start+SAMPLE_LENGTH]]
    if mode == 1:
        sample[int(rng.integers(0, SAMPLE_LENGTH))] = None
    elif mode == 3:
        # A three-year observed subsequence gives at most two paired changes.
        first = int(rng.integers(0, SAMPLE_LENGTH-2))
        sample = [w if first <= i < first+3 else None for i, w in enumerate(sample)]
    elif mode == 4:
        sample = rng.integers(1, 230, size=SAMPLE_LENGTH).tolist()
    elif mode == 5:
        j = int(rng.integers(0, SAMPLE_LENGTH))
        sample[j] = min(255, max(1, sample[j] + int(rng.choice([-35, 35]))))
    elif mode == 6:
        for i in rng.choice(size, size=max(1, size // 7), replace=False):
            master[int(i)] = None
    return {"reference": master, "sample": sample}
