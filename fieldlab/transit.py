"""Search a synthetic brightness series for a periodic, one-cadence dip.

Cadences are zero-indexed. Trial periods are 3, 4, 5, 7 and 9 cadences.
For a trial (period, phase), use the observed samples at that phase. Score
the sum of (baseline - flux - 4). This equals the reduction in squared error
from a fixed depth-eight dip relative to the known flat baseline, divided
by 16. Require a score of at least 12 and at least three samples at least
five flux units below baseline. Maximize score, then prefer the earlier
period in the bank and the smaller phase. Null samples contribute nothing.

Phase queries use softmax attention over all cadence tokens. A programmed
quadratic address selects equal residues of time modulo the trial period.
The value projection carries dip evidence and threshold support. Multiplying
the attention mean by the number of scheduled phase visits recovers sums.
The classifier applies the support requirement, score cutoff and tie rule.
This single cross-attention block searches the whole bank in parallel.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PERIODS = (3, 4, 5, 7, 9)
TEMPLATES = tuple((period, phase) for period in PERIODS for phase in range(period))
CLASSES = len(TEMPLATES) + 1
MAX_CADENCES = 192


def encode(record):
    """Provide the baseline, measured flux and an observation bit at each cadence."""
    if not isinstance(record, dict):
        raise ValueError("Expected a brightness-series object")
    base = record.get("baseline")
    flux = record.get("flux")
    if type(base) is not int or not 16 <= base <= 4095:
        raise ValueError("baseline must be an integer from 16 through 4095")
    if not isinstance(flux, list) or not 1 <= len(flux) <= MAX_CADENCES:
        raise ValueError("flux must be a list of 1 through 192 samples")
    if any(v is not None and (type(v) is not int or abs(v - base) > 16) for v in flux):
        raise ValueError("Each observed flux must be an integer within 16 units of baseline")
    return np.asarray([(base, base if v is None else v, int(v is not None)) for v in flux], dtype=np.int64)


class TransitTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # For residue r and trial phase q, the logit is 128*(2*q*r-r*r).
        self.address = nn.Parameter(torch.tensor([256., -128.]), requires_grad=False)
        self.thresholds = nn.Parameter(torch.tensor([4., 5.]), requires_grad=False)
        self.value = nn.Parameter(torch.tensor([[1., 0.], [-4., 0.], [0., 1.], [0., -1.]]),
                                  requires_grad=False)
        self.register_buffer("period_phase", torch.tensor(TEMPLATES, dtype=torch.long), persistent=False)

    def forward(self, tokens):
        baseline, flux, observed = tokens.unbind(dim=-1)
        deficit = ((baseline - flux) * observed).to(self.value.dtype)
        # For integer deficits, ReLU(d-4) - ReLU(d-5) is the support bit.
        steps = F.relu(deficit.unsqueeze(-1) - self.thresholds)
        values = torch.cat((deficit.unsqueeze(-1), observed.unsqueeze(-1).to(deficit.dtype), steps), dim=-1)
        values = values @ self.value
        periods, phases = self.period_phase.unbind(-1)
        length = tokens.shape[1]
        times = torch.arange(length, device=tokens.device)
        residue = (times[None, :] % periods[:, None]).to(deficit.dtype)
        keys = torch.stack((residue, residue.square()), dim=-1)
        q = phases.to(deficit.dtype)
        queries = torch.stack((q, torch.ones_like(q)), dim=-1) * self.address
        attention = torch.bmm(queries[:, None, :], keys.transpose(1, 2)).squeeze(1).softmax(-1)
        visits = ((length - 1 - phases) // periods + 1).clamp_min(0)
        totals = (attention @ values) * visits[None, :, None]
        evidence, support = totals.unbind(-1)
        # With two support events, the greatest possible sum is 24. The
        # penalty is at least 64 whenever fewer than three events qualify.
        candidate = evidence - 128 * F.relu(2.5 - support)
        tie = torch.arange(1, CLASSES, device=tokens.device, dtype=deficit.dtype) / 128
        # Eligible integer scores >=12 exceed 11.5 even after the tie offset.
        no_signal = deficit.new_full((tokens.shape[0], 1), 11.5)
        return torch.cat((no_signal, candidate - tie), dim=1).unsqueeze(1)


def build_model():
    return TransitTransformer().eval()


def reference(record):
    """Enumerate individual samples and compare exact integer scores."""
    encode(record)
    baseline, flux = record["baseline"], record["flux"]
    best_label, best_score = 0, 11
    for label, (period, phase) in enumerate(TEMPLATES, start=1):
        seen = [baseline - flux[i] for i in range(phase, len(flux), period) if flux[i] is not None]
        score = sum(dip - 4 for dip in seen)
        if sum(dip >= 5 for dip in seen) >= 3 and score > best_score:
            best_label, best_score = label, score
    return np.asarray([best_label], dtype=np.int64)


def example():
    flux = [100, 101, 92, 100, 99, 102, 100, 92, 100, 99, 100, 100,
            None, 100, 101, 99, 100, 92, 102, 100, 99, 100, 92, 100]
    return {"baseline": 100, "flux": flux}


def generate(rng, size, index):
    """Plant a repeated dip or generate a control with noise, gaps and transient dimming."""
    if size not in (24, 48, 96, 192):
        raise ValueError("Benchmark series have 24, 48, 96 or 192 cadences")
    baseline = int(rng.integers(700, 1101))
    flux = baseline + rng.integers(-2, 3, size=size)
    missing = rng.random(size) < 0.1
    t = np.arange(size)
    variant = index % 8
    planted = None
    if variant % 2 == 0:
        period = int(rng.choice(PERIODS))
        if variant == 4:
            period = 9
        phase = int(rng.integers(period))
        planted = {"period": period, "phase": phase}
        transit = t % period == phase
        depth = 6 if variant == 2 else 8
        flux[transit] -= depth
        if variant == 4:
            # Remove precisely the visits that distinguish periods three and nine.
            missing |= ((t % 3) == (phase % 3)) & ~transit
        elif variant == 6:
            start = int(rng.integers(size - size // 4 + 1))
            missing[start:start + size // 4] = True
            flux[int(rng.integers(size))] -= 6
    elif variant == 3:
        flux[int(rng.integers(size))] -= 13
    elif variant == 5:
        transients = rng.choice(size, size=max(3, size // 10), replace=False)
        flux[transients] -= rng.integers(7, 13, size=len(transients))
    elif variant == 7:
        start = int(rng.integers(size - 3))
        flux[start:start + 4] -= np.array([6, 12, 10, 7])
    measured = [None if masked else int(v) for v, masked in zip(flux, missing)]
    return {"baseline": baseline, "flux": measured, "injected": planted}


def format_output(record, prediction):
    encode(record)
    if len(prediction) != 1 or not 0 <= prediction[0] < CLASSES:
        raise ValueError("Expected one class from 0 through 28")
    if prediction[0] == 0:
        return {"period": None, "phase": None}
    period, phase = TEMPLATES[prediction[0] - 1]
    return {"period": period, "phase": phase}
