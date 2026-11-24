"""Fixed, feedforward GPU comparators for inventory, bookings and alarm states.

Inputs use the public applications' integer token schemas. Each call emits one
floating-point score per item; score > 0 means flagged. Full variants can see
the entire permitted context. Local variants replace it with 2, 8 or 32 taps.
There is no training, per-item Python loop, or recurrent evaluation.

Parameter counts include every entry (including zeros) in registered floating
coefficient tensors. Source-level masks, structural indices, padding, logical
operators and fixed label conventions are described separately in metadata.
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


WINDOWS = (2, 8, 32)
MAX_SIZE = 128


class Feedforward(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.roles = {}

    def coefficient(self, name, values, purpose):
        value = torch.as_tensor(values, dtype=torch.float64)
        self.register_parameter(name, nn.Parameter(value, requires_grad=False))
        self.roles[name] = purpose

    def metadata(self):
        coefficients = {name: {"shape": list(p.shape), "entries": p.numel(),
                               "nonzero": int(torch.count_nonzero(p).item()), "purpose": self.roles[name]}
                        for name, p in self.named_parameters()}
        return {"name": "full_feedforward" if self.window is None else f"local_w{self.window}",
                "task": self.task, "architecture": self.architecture, "window": self.window,
                "context": self.context, "fixed_operations": self.operations,
                "parameter_definition": "All entries of registered floating coefficients, including zeros. Masks, index arithmetic, padding and logical operations are fixed code.",
                "parameters": sum(v["entries"] for v in coefficients.values()),
                "nonzero_parameters": sum(v["nonzero"] for v in coefficients.values()),
                "trainable_parameters": 0, "coefficients": coefficients}


def _causal_mask(n, device):
    row = torch.arange(n, device=device)
    return row[None, :] > row[:, None]


class Inventory(Feedforward):
    task = "inventory"

    def __init__(self, window):
        super().__init__(window)
        self.coefficient("offsets", [16, 25], "Decode signed quantity and first-event opening-balance IDs")
        self.coefficient("margin", 0.5, "Half-unit class boundary between a negative integer and zero")
        if window is None:
            self.architecture = "One uniform causal attention head and a signed readout"
            self.context = "All transactions at or before each event, including the first opening balance"
            self.operations = "Mask future rows, apply softmax to zero logits, average the signed inputs and scale the class margin by prefix length. ID zero means no opening entry."
        else:
            self.coefficient("taps", [[1] * window], "One fixed signed-quantity coefficient per causal lag")
            self.architecture = "One causal finite-impulse-response filter and a signed readout"
            self.context = f"Current event plus at most {window - 1} earlier ones; the opening balance appears only in the first event"
            self.operations = "Left zero-padding, extraction of consecutive windows, and a dense dot product with the registered taps. ID zero means no opening entry."

    def forward(self, tokens):
        n = tokens.shape[1]
        qty = tokens[..., 1].to(self.offsets.dtype) - self.offsets[0]
        opening = tokens[..., 2].to(self.offsets.dtype)
        qty = qty + torch.where(opening != 0, opening - self.offsets[1], 0)
        if self.window is not None:
            block = F.pad(qty, (self.window - 1, 0)).unfold(-1, self.window, 1)
            return -F.linear(block, self.taps).squeeze(-1) - self.margin
        scores = qty.new_zeros((n, n)).masked_fill(_causal_mask(n, qty.device), -torch.inf)
        weights = scores.softmax(dim=-1)
        mean = (weights @ qty.unsqueeze(-1)).squeeze(-1)
        length = torch.arange(1, n + 1, dtype=qty.dtype, device=qty.device)
        return -mean - self.margin / length


class Telemetry(Feedforward):
    task = "telemetry"

    def __init__(self, window):
        super().__init__(window)
        self.coefficient("thresholds", [30, 70], "Inclusive alarm reset and trip thresholds")
        # Four logit units per timestamp separate the newest write from every
        # older one. Their geometric tail has less mass than the newest write.
        self.coefficient("write_bonus", 4 * (window or MAX_SIZE), "Score bonus for a reset or trip over a hold")
        if window is None:
            self.coefficient("recency", 4, "Logit gain per raw reading position")
            self.architecture = "One causal attention head over thresholded writes"
            self.context = "Every preceding reading and the current one"
            self.operations = "Threshold comparisons identify writes, position supplies recency, and causal softmax pools +1 for a trip and -1 otherwise. A prefix of holds votes off."
        else:
            self.coefficient("lag_scores", [4 * (i + 1) for i in range(window)], "One active attention logit per causal lag")
            self.architecture = "One local causal attention head with a separate bias per lag"
            self.context = f"Current reading plus at most {window - 1} preceding readings"
            self.operations = "Threshold comparisons, left padding with the earliest reading, sliding-window softmax and +1/-1 votes. Repeated left-edge readings preserve the sign on short prefixes."

    def forward(self, tokens):
        n = tokens.shape[1]
        readings = tokens[..., 0].to(self.thresholds.dtype)
        if self.window is None:
            trip = readings >= self.thresholds[1]
            decisive = trip | (readings <= self.thresholds[0])
            keys = tokens[..., 1].to(readings.dtype) * self.recency + decisive.to(readings.dtype) * self.write_bonus
            scores = keys[:, None, :].expand(-1, n, -1).masked_fill(_causal_mask(n, readings.device), -torch.inf)
            votes = torch.where(trip, torch.ones_like(readings), -torch.ones_like(readings))
            return (scores.softmax(dim=-1) @ votes.unsqueeze(-1)).squeeze(-1)
        first = readings[:, :1].expand(-1, self.window - 1)
        block = torch.cat((first, readings), dim=1).unfold(-1, self.window, 1)
        trip = block >= self.thresholds[1]
        decisive = trip | (block <= self.thresholds[0])
        scores = self.lag_scores + decisive.to(block.dtype) * self.write_bonus
        votes = torch.where(trip, torch.ones_like(block), -torch.ones_like(block))
        return (scores.softmax(dim=-1) * votes).sum(dim=-1)


class Scheduling(Feedforward):
    task = "scheduling"

    def __init__(self, window):
        super().__init__(window)
        # For pair [start_i,end_i,start_j,end_j], compute a=end_i-start_j,
        # b=end_j-start_i. Three hidden units express min(a,b) as
        # ReLU(a) - ReLU(-a) - ReLU(a-b), the strict-overlap signed margin.
        self.coefficient("hidden", [[0, 1, -1, 0], [0, -1, 1, 0], [1, 1, -1, -1]],
                         "Dense 3-by-4 linear map from endpoint pairs to a, -a and a-b")
        self.coefficient("projection", [[1, -1, -1]], "Combine the rectified pair features into the intersection predicate")
        self.coefficient("margin", 0.5, "Half-minute class boundary between touching and overlapping")
        if window is None:
            self.architecture = "Shared pairwise ReLU network followed by masked maximum over all rows"
            self.context = "Every other booking, in either direction in the submitted record"
            self.operations = "Construct endpoint pairs, exclude self by its array index, then take a row maximum. The neutral self score is -1."
        else:
            self.coefficient("taps", [[1] * window], "One active coefficient per distinct neighbor offset")
            self.architecture = "Shared pairwise ReLU network followed by a local fixed-weight pooling layer"
            self.context = f"Up to {window // 2} earlier and {window // 2} later bookings in the supplied order; self excluded"
            self.operations = "Zero-length [0,0) boundary padding, pair extraction at each nonzero offset, clip pair margins to [0,1], and pool through registered taps."

    def forward(self, tokens):
        spans = tokens[..., :2].to(self.hidden.dtype)
        b, n, _ = spans.shape
        if self.window is None:
            neighbors = spans[:, None, :, :].expand(b, n, n, 2)
        else:
            r = self.window // 2
            padded = F.pad(spans.transpose(1, 2), (r, r))
            centered = padded.unfold(-1, self.window + 1, 1).permute(0, 2, 3, 1)
            neighbors = torch.cat((centered[:, :, :r], centered[:, :, r+1:]), dim=2)
        query = spans[:, :, None, :].expand_as(neighbors)
        pair = torch.cat((query, neighbors), dim=-1)
        signed = F.linear(F.relu(F.linear(pair, self.hidden)), self.projection).squeeze(-1)
        if self.window is not None:
            return F.linear(signed.clamp(0, 1), self.taps).squeeze(-1) - self.margin
        self_pair = torch.eye(n, dtype=torch.bool, device=spans.device)
        return signed.masked_fill(self_pair, -1).amax(dim=-1) - self.margin


def build_baselines(task):
    """Return four independent analytic models; all floating coefficients start FP64."""
    model = {"inventory": Inventory, "scheduling": Scheduling, "telemetry": Telemetry}[task]
    variants = OrderedDict(full_feedforward=model(None))
    variants.update((f"local_w{window}", model(window)) for window in WINDOWS)
    return variants


def baseline_metadata(model):
    return model.metadata()
