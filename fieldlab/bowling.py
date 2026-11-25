"""Programmed attention scorekeeper for scratch ten-pin bowling.

A record groups credited pinfall by frame. A scorecard can stop after any
delivery, including before either fill ball in frame ten. Its cumulative
score stays blank until all deliveries needed to determine it have occurred.
Only started frames are emitted. Splits score as the pin counts supplied.
The input represents credited pinfall, without foul or pin-location notation.
Rules and examples: https://bowl.com/keeping-score

Two attention heads read the first two observed deliveries following each
frame. The query position comes from counting occupied input slots. Added
sentinel tokens make a missing future delivery distinguishable from a gutter
ball. Frame-local gates apply the ordinary strike/spare and tenth-frame
rules. A tied prefix block gathers earlier frame contributions at successive
power-of-two offsets, accumulating both points and unresolved frames. The
score/status readout has 302 classes: 0..300 or 301 for pending.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PENDING = 301


def encode(record):
    frames = record.get("frames") if isinstance(record, dict) else None
    if not isinstance(frames, list) or not 1 <= len(frames) <= 10:
        raise ValueError("Supply one through ten started frames")
    tokens = np.full((len(frames), 3), -1, dtype=np.int64)
    for i, rolls in enumerate(frames):
        if (not isinstance(rolls, list) or not 1 <= len(rolls) <= 3
                or any(type(x) is not int or not 0 <= x <= 10 for x in rolls)):
            raise ValueError("A frame holds one through three credited pin counts, each from 0 through 10")
        strike = rolls[0] == 10
        if len(rolls) >= 2 and not strike and rolls[0] + rolls[1] > 10:
            raise ValueError("The second delivery exceeds the remaining pins")
        if i < 9:
            needed = 1 if strike else 2
            if len(rolls) > needed or (i < len(frames)-1 and len(rolls) != needed):
                raise ValueError("Frame must end after a strike or two deliveries before starting the next frame")
        elif len(rolls) == 3:
            if not strike and rolls[0] + rolls[1] < 10:
                raise ValueError("An open tenth frame has no fill ball")
            if strike and rolls[1] < 10 and rolls[1] + rolls[2] > 10:
                raise ValueError("The last delivery exceeds the pins left on the fill rack")
        tokens[i, :len(rolls)] = rolls
    return tokens


class PositionAttention(nn.Module):
    """Soft lookup at an integer delivery or frame position with fixed key features."""
    def __init__(self):
        super().__init__()
        # Dot product of [128*q, -64] and [k, k²] has unique maximum at k=q.
        self.projection = nn.Parameter(torch.tensor([128., -64.]), requires_grad=False)

    def forward(self, queries, keys, values, live=None):
        q = torch.stack((queries, torch.ones_like(queries)), dim=-1) * self.projection
        k = torch.stack((keys, keys.square()), dim=-1)
        logits = q @ k.transpose(-1, -2)
        if live is not None:
            logits = logits.masked_fill(~live[:, None, :], -torch.inf)
        return logits.softmax(-1) @ values


class Scorekeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = PositionAttention()
        self.own_pin_weights = nn.Parameter(torch.ones(3), requires_grad=False)
        self.status_weight = nn.Parameter(torch.tensor([2., -1.]), requires_grad=False)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 3 or not 1 <= tokens.shape[1] <= 10:
            raise ValueError("Expected one through ten three-slot frame tokens per record")
        batch, n, _ = tokens.shape
        dtype = self.own_pin_weights.dtype
        device = tokens.device
        live = tokens >= 0
        pins = tokens.clamp_min(0).to(dtype)
        count = live.sum(-1).to(dtype)
        ends = count.cumsum(-1)
        flat_live = live.flatten(1)
        ordinals = flat_live.cumsum(-1).to(dtype)
        offsets = torch.arange(1, 3, device=device, dtype=dtype)
        # Beyond the end of the current game prefix, two zero-valued sentinel
        # deliveries have a presence flag of zero. A real gutter ball has one.
        keys = torch.cat((ordinals, ends[:, -1:] + offsets), dim=1)
        kv = torch.stack((pins.flatten(1), flat_live.to(dtype)), dim=-1)
        kv = F.pad(kv, (0, 0, 0, 2))
        queries = (ends[:, :, None] + offsets).reshape(batch, 2*n)
        lookahead = self.position(queries, keys, kv, F.pad(flat_live, (0, 2), value=True))
        lookahead = lookahead.reshape(batch, n, 2, 2)

        # A mark earns the first following ball; a strike also earns the second.
        # Frame ten already encodes its fill deliveries inside its own token.
        regular = (torch.arange(n, device=device) < 9).to(dtype)
        strike = F.relu(pins[:, :, 0] - 9)
        marked = (pins[:, :, :2].sum(-1) - 9).clamp(0, 1)
        bonus = torch.stack((marked, strike), dim=-1) * regular[:, None]
        score = pins @ self.own_pin_weights + (bonus * lookahead[:, :, :, 0]).sum(-1)
        needed = regular*(2 - strike) + (1 - regular)*(2 + marked)
        missing = F.relu(needed - count) + (bonus * (1 - lookahead[:, :, :, 1])).sum(-1)
        state = torch.stack((score, missing), dim=-1)

        # Logarithmic-depth inclusive scan. Frame zero is a BOS token carrying
        # zero points and zero outstanding deliveries.
        frame_keys = torch.arange(n + 1, device=device, dtype=dtype).unsqueeze(0)
        for step in range((n - 1).bit_length()):
            predecessor = (frame_keys[:, 1:] - (1 << step)).clamp_min(0)
            state = state + self.position(predecessor, frame_keys, F.pad(state, (0, 0, 1, 0)))
        classes = torch.arange(PENDING, device=device, dtype=dtype)
        score_logits = -(state[:, :, 0:1] - classes).abs()
        pending = state[:, :, 1:2].clamp(0, 1)
        pending_logit = pending*self.status_weight[0] + self.status_weight[1]
        return torch.cat((score_logits, pending_logit), dim=-1)


def build_model():
    return Scorekeeper().eval()


def reference(record):
    """Grade frames directly by reading literal rolls after each frame."""
    encode(record)
    frames = record["frames"]
    cumulative = 0
    out = []
    for i, frame in enumerate(frames):
        own = sum(frame)
        strike = frame[0] == 10
        if i == 9:
            award = strike or (len(frame) >= 2 and sum(frame[:2]) == 10)
            known = len(frame) == 2 + int(award)
            points = own
        else:
            following = [v for later in frames[i+1:] for v in later]
            bonus = 2 if strike else int(own == 10)
            known = (strike or len(frame) == 2) and len(following) >= bonus
            points = own + sum(following[:bonus])
        if not known or cumulative == PENDING:
            cumulative = PENDING
        else:
            cumulative += points
        out.append(cumulative)
    return np.asarray(out, dtype=np.int64)


def _regular(rng, kind):
    if kind == 2:
        return [10]
    first = int(rng.integers(10))
    second = 10-first if kind == 1 else int(rng.integers(10-first))
    return [first, second]


def generate(rng, size, index):
    if not 1 <= size <= 10:
        raise ValueError("Use one through ten started frames")
    frames = [_regular(rng, int(rng.integers(3))) for _ in range(size - 1)]
    if size >= 3 and index % 5 == 0:
        frames[-2:] = [[10], [10]]
    mode = index % 8
    if size < 10:
        last = _regular(rng, mode % 3)
        if mode == 4:
            last = [int(rng.integers(10))]
    elif mode == 0:
        last = _regular(rng, 0)
    elif mode == 1:
        last = [int(rng.integers(10))]
    elif mode == 2:
        last = _regular(rng, 1)
    elif mode == 3:
        last = [10, int(rng.integers(10))]
    elif mode == 4:
        last = [10, 10]
    elif mode == 5:
        last = _regular(rng, 1) + [int(rng.integers(11))]
    elif mode == 6:
        fill = int(rng.integers(10))
        last = [10, fill, int(rng.integers(11-fill))]
    else:
        last = [10, 10, int(rng.integers(11))]
    return {"frames": frames + [last]}


def example():
    return {"frames": [[10], [10], [7, 3], [9, 0], [10]]}


def format_output(record, prediction):
    if len(prediction) != len(record["frames"]):
        raise ValueError("Expected one score for every started frame")
    return {"cumulative": [None if value == PENDING else int(value) for value in prediction]}
