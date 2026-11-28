"""Add nominal chlorine/bromine isotope envelopes with a shared attention block."""

from math import comb, log

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MZ_MIN = 400
MZ_MAX = 415
MAX_IONS = 16
MAX_ATOMS = 3
UNITS = 4**MAX_ATOMS
FIELDS = ("mz", "cl", "br", "abundance")


def encode(record):
    if not isinstance(record, dict) or set(record) != {"ions"}:
        raise ValueError("Supply an ions list")
    ions = record["ions"]
    if not isinstance(ions, list) or not 1 <= len(ions) <= MAX_IONS:
        raise ValueError("Supply 1 to 16 ion components")
    rows = []
    for ion in ions:
        if (
            not isinstance(ion, dict)
            or set(ion) != set(FIELDS)
            or any(type(ion[name]) is not int for name in FIELDS)
        ):
            raise ValueError("Each component requires integer mz, cl, br and abundance")
        mz, cl, br, amount = (ion[name] for name in FIELDS)
        if not MZ_MIN <= mz <= MZ_MAX - 2 * MAX_ATOMS or not 1 <= amount <= 3:
            raise ValueError("Use light-isotope mz 400–409 and abundance 1–3")
        if min(cl, br) < 0 or cl + br > MAX_ATOMS:
            raise ValueError("Each ion may contain zero through three halogen atoms in total")
        rows.append((mz, cl, br, amount))
    # A spectrum is unchanged when the input list is permuted.
    return np.array(sorted(rows), dtype=np.int64)


class IsotopeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Log odds for heavy versus light: padding, chlorine, bromine.
        self.heavy_keys = nn.Parameter(torch.tensor([-128.0, -log(3.0), 0.0]), requires_grad=False)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.size(-1) != 4:
            raise ValueError("Expected a batch of component records")
        mz, cl, br, amount = tokens.to(self.heavy_keys.dtype).unbind(-1)
        bins = torch.arange(MZ_MIN, MZ_MAX + 1, dtype=mz.dtype, device=mz.device)
        mass = UNITS * amount.unsqueeze(-1) * F.relu(1 - (mz.unsqueeze(-1) - bins).abs())
        keys = torch.stack((torch.zeros_like(self.heavy_keys), self.heavy_keys), dim=-1)
        for atom in range(MAX_ATOMS):
            query = torch.stack((atom >= cl + br, atom < cl, (cl <= atom) & (atom < cl + br)), dim=-1)
            weights = (query.to(keys.dtype) @ keys).softmax(-1)
            # Every output mass bin reads its current intensity and the intensity
            # two bins to its left. A heavy substitution shifts that mass up two.
            shifted = F.pad(mass[..., :-2], (2, 0))
            values = torch.stack((mass, shifted), dim=-2)
            mass = (weights.unsqueeze(-2) @ values).squeeze(-2)
        spectrum = mass.sum(dim=1)
        classes = torch.arange(UNITS * 3 * MAX_IONS + 1, dtype=mz.dtype, device=mz.device)
        return -(spectrum.unsqueeze(-1) - classes).abs()


def build_model():
    return IsotopeTransformer().eval()


def reference(record):
    """Use binomial multiplicities directly for both elements."""
    ions = encode(record)
    spectrum = [0] * (MZ_MAX - MZ_MIN + 1)
    for fields in ions:
        mz, cl, br, amount = map(int, fields)
        multiplier = amount * 4 ** (MAX_ATOMS - cl - br) * 2**br
        for hc in range(cl + 1):
            for hb in range(br + 1):
                abundance = multiplier * comb(cl, hc) * 3 ** (cl - hc) * comb(br, hb)
                spectrum[mz - MZ_MIN + 2 * (hc + hb)] += abundance
    return np.asarray(spectrum, dtype=np.int64)


def generate(rng, size, index):
    if not 1 <= size <= MAX_IONS:
        raise ValueError("Invalid mixture size")
    # Uniform parents, same-parity clusters or a narrow band of parent masses.
    # Same-parity and co-located parents increase overlaps of isotope branches.
    if index % 3 == 0:
        locations = np.arange(400, 410)
    elif index % 3 == 1:
        locations = np.arange(400 + int(rng.integers(2)), 410, 2)
    else:
        start = int(rng.integers(400, 408))
        locations = np.arange(start, start + 3)
    formulas = [(c, b) for c in range(MAX_ATOMS + 1) for b in range(MAX_ATOMS + 1 - c)]
    rows = []
    for _ in range(size):
        cl, br = formulas[int(rng.integers(len(formulas)))]
        rows.append((int(rng.choice(locations)), cl, br, int(rng.integers(1, 4))))
    return {"ions": [dict(zip(FIELDS, row)) for row in sorted(rows)]}


def example():
    return {
        "ions": [
            {"mz": 400, "cl": 2, "br": 0, "abundance": 1},
            {"mz": 401, "cl": 0, "br": 0, "abundance": 1},
            {"mz": 402, "cl": 0, "br": 1, "abundance": 2},
        ]
    }


def format_output(record, prediction):
    if len(prediction) != MZ_MAX - MZ_MIN + 1:
        raise ValueError("Expected one intensity per nominal mass bin")
    return {"mz_min": MZ_MIN, "intensities": list(prediction)}
