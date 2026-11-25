"""Draw the upper face of a rising-shed shaft-loom draft with two attention reads."""

from itertools import groupby

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SHAFTS = TREADLES = 8
MAX_ENDS = MAX_PICKS = 64


def _id_list(values, limit, field):
    if (not isinstance(values, list) or
            any(type(v) is not int or not 1 <= v <= limit for v in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"{field} must list distinct integers from 1 through {limit}")


def encode(record):
    """Serialize the three original draft charts into separate portions of one array."""
    if not isinstance(record, dict) or set(record) != {"threading", "tie_up", "treadling"}:
        raise ValueError("Supply threading, tie_up and treadling")
    ends, ties, picks = record["threading"], record["tie_up"], record["treadling"]
    if (not isinstance(ends, list) or not 1 <= len(ends) <= MAX_ENDS or
            any(type(shaft) is not int or not 1 <= shaft <= SHAFTS for shaft in ends)):
        raise ValueError("Supply 1–64 warp ends, each threaded on shaft 1–8")
    if not isinstance(ties, list) or not 1 <= len(ties) <= TREADLES:
        raise ValueError("Supply the tie-up for 1–8 treadles")
    if not isinstance(picks, list) or not 1 <= len(picks) <= MAX_PICKS:
        raise ValueError("Supply 1–64 treadling rows")

    tokens = np.zeros((TREADLES + len(picks), SHAFTS + len(ends)), dtype=np.int64)
    tokens[0, SHAFTS:] = ends
    for pedal, shafts in enumerate(ties):
        _id_list(shafts, SHAFTS, "A tie-up row")
        for shaft in shafts:
            tokens[pedal, shaft - 1] = 1
    for row, pedals in enumerate(picks, start=TREADLES):
        _id_list(pedals, len(ties), "A treadling row")
        for pedal in pedals:
            tokens[row, pedal - 1] = 1
    return tokens


class LoomTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shaft_gain = nn.Parameter(torch.tensor(32.), requires_grad=False)
        self.active_gain = nn.Parameter(torch.tensor(64.), requires_grad=False)
        self.lift_gain = nn.Parameter(torch.tensor(16.), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, tokens):
        ends = tokens[:, 0, SHAFTS:] - 1
        tie = tokens[:, :TREADLES, :SHAFTS].to(self.shaft_gain.dtype)
        # One-hot shaft address queries and identity keys read the connections
        # between this warp end's shaft and each treadle.
        queries = F.one_hot(ends, SHAFTS).to(tie.dtype) * self.shaft_gain
        lifts = queries.softmax(-1) @ tie.transpose(1, 2)
        # A neutral treadle is always eligible and lifts nothing. It lets an
        # empty pick use exactly the same attention computation as any other.
        lifts = F.pad(lifts, (0, 1))
        active = F.pad(tokens[:, TREADLES:, :SHAFTS].to(tie.dtype), (0, 1), value=1.)
        # Keys carry the treadle identity and its lift value at this warp end.
        # The query strongly prefers active pedals, then those that lift it.
        scores = active[:, :, None, :] * self.active_gain + lifts[:, None, :, :] * self.lift_gain
        visible = (scores.softmax(-1) * lifts[:, None, :, :]).sum(-1)
        margin = visible.flatten(1) - self.threshold
        return torch.stack((-margin, margin), dim=-1)


def build_model():
    return LoomTransformer().eval()


def reference(record):
    """Construct the moving shaft set for each shed, then inspect every warp end."""
    result = []
    for pedals in record["treadling"]:
        raised = {shaft for pedal in pedals for shaft in record["tie_up"][pedal - 1]}
        result.extend(int(shaft in raised) for shaft in record["threading"])
    return np.asarray(result, dtype=np.int64)


def example():
    # Two-over, two-under twill on four shafts.
    return {"threading": [1, 2, 3, 4] * 2, "tie_up": [[1, 2], [2, 3], [3, 4], [4, 1]],
            "treadling": [[1], [2], [3], [4]] * 2}


def format_output(record, prediction):
    ends, picks = len(record["threading"]), len(record["treadling"])
    if len(prediction) != ends * picks or any(x not in (0, 1) for x in prediction):
        raise ValueError("Expected one binary crossing per end and pick")
    cloth = ["".join("#" if x else "." for x in prediction[i:i + ends])
             for i in range(0, len(prediction), ends)]

    def longest(lines, symbol):
        return max((sum(1 for _ in run) for line in lines for face, run in groupby(line)
                    if face == symbol), default=0)

    return {"drawdown": cloth, "longest_warp_float_picks": longest(zip(*cloth), "#"),
            "longest_weft_float_ends": longest(cloth, ".")}


def generate(rng, size, index):
    shafts = int(rng.integers(2, SHAFTS + 1))
    shaft_order = rng.permutation(np.arange(1, shafts + 1)).tolist()
    offset = int(rng.integers(shafts))
    kinds = index % 4
    if kinds in (0, 1):
        # Straight and point threadings with cyclic tie-ups. A point changes
        # direction without doubling either endpoint.
        cycle = shaft_order if kinds == 0 else shaft_order + shaft_order[-2:0:-1]
        ends = [cycle[(i + offset) % len(cycle)] for i in range(size)]
        width = int(rng.integers(1, shafts))
        tie = [[shaft_order[(t + k) % shafts] for k in range(width)] for t in range(shafts)]
        pedal_order = list(range(1, shafts + 1))
        if kinds == 1:
            pedal_order += pedal_order[-2:0:-1]
        phase = int(rng.integers(len(pedal_order)))
        treadling = [[pedal_order[(i + phase) % len(pedal_order)]] for i in range(size)]
    else:
        n_pedals = int(rng.integers(2, TREADLES + 1))
        ends = rng.choice(shaft_order, size=size).tolist()
        tie = [rng.choice(shaft_order, size=int(rng.integers(0, shafts + 1)),
                          replace=False).tolist() for _ in range(n_pedals)]
        pedal_order = np.arange(1, n_pedals + 1)
        treadling = [rng.choice(pedal_order, size=int(rng.integers(0, min(3, n_pedals) + 1)),
                                replace=False).tolist() for _ in range(size)]
        if kinds == 3:
            # Include an idle pick and a simultaneous multi-pedal lift.
            treadling[0], treadling[1] = [], rng.choice(pedal_order, size=2, replace=False).tolist()
    return {"threading": ends, "tie_up": tie, "treadling": treadling}
