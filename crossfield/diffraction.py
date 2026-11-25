"""Exact kinematical structure-factor intensities for quarter-grid point motifs.

Each site contains three fractional-coordinate numerators over 4 and a positive
integer scattering amplitude. For reflection H the forward model returns
|sum_j a_j exp(2*pi*i*H.x_j/4)|^2. The International Union of Crystallography
defines this structure factor at https://dictionary.iucr.org/Structure_factor .
Constant real amplitudes and discrete sites specify the synthetic model.
A zero for an individual motif can arise from cancellation between its sites.
Systematic absences attributed to a space group require its symmetry conditions.

Cross-attention uses the input Miller indices as queries. Each site supplies a
bank of keys indexed by possible integer phases m. Scores have the form
128*(2*m*H.x - m^2), so the peak occurs at m=H.x. The associated complex
fourth root of unity is the value. A second, uniform attention pool sums
the weighted contributions across sites. A ReLU spline squares the real and
imaginary parts on their integer support before the intensity-class readout.
"""

import cmath
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MAX_SITES = 8
MAX_REFLECTIONS = 32
MAX_INDEX = 4
MAX_COMPONENT = 2 * MAX_SITES
CLASSES = MAX_COMPONENT ** 2 + 1
FCC = ((0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0))


def encode(record):
    """Serialize the literal inputs, padding unused site amplitudes with zero."""
    if not isinstance(record, dict):
        raise ValueError("Expected a crystal record")
    sites, refs = record.get("sites"), record.get("reflections")
    if not isinstance(sites, list) or not 1 <= len(sites) <= MAX_SITES:
        raise ValueError("Supply 1 through 8 point sites")
    if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_REFLECTIONS:
        raise ValueError("Supply 1 through 32 Miller triples")
    tokens = np.zeros((MAX_SITES + len(refs), 4), dtype=np.int64)
    positions = set()
    for j, site in enumerate(sites):
        if (not isinstance(site, list) or len(site) != 4 or any(type(v) is not int for v in site)
                or any(not 0 <= v <= 3 for v in site[:3]) or site[3] not in (1, 2)):
            raise ValueError("Site must be [x,y,z,amplitude] with x,y,z in 0..3 and amplitude 1 or 2")
        xyz = tuple(site[:3])
        if xyz in positions:
            raise ValueError("A point-site position must occur only once")
        positions.add(xyz)
        tokens[j] = site
    for j, hkl in enumerate(refs):
        if (not isinstance(hkl, list) or len(hkl) != 3
                or any(type(v) is not int or abs(v) > MAX_INDEX for v in hkl)):
            raise ValueError("A Miller triple must contain three integers from -4 to 4")
        tokens[MAX_SITES + j, :3] = hkl
    return tokens


class DiffractionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.address = nn.Parameter(torch.tensor([256., -128.]), requires_grad=False)
        self.roots = nn.Parameter(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), requires_grad=False)
        self.pool_gain = nn.Parameter(torch.tensor(float(MAX_SITES)), requires_grad=False)
        self.knots = nn.Parameter(torch.arange(MAX_COMPONENT, dtype=torch.float32), requires_grad=False)
        slopes = torch.full((MAX_COMPONENT,), 2.)
        slopes[0] = 1.
        self.slopes = nn.Parameter(slopes, requires_grad=False)

    def forward(self, tokens):
        if (tokens.ndim != 3 or tokens.shape[-1] != 4
                or not 1 <= tokens.shape[1] - MAX_SITES <= MAX_REFLECTIONS):
            raise ValueError("Expected eight site tokens followed by the reflection tokens")
        raw = tokens.to(self.roots.dtype)
        xyz, strength = raw[:, :MAX_SITES, :3], raw[:, :MAX_SITES, 3]
        hkl = raw[:, MAX_SITES:, :3]
        max_phase = 3 * MAX_INDEX * 3
        m = torch.arange(-max_phase, max_phase + 1, device=tokens.device)
        phase = m.to(self.roots.dtype)
        linear = xyz.unsqueeze(-2) * phase[:, None]
        quadratic = phase.square().expand(*linear.shape[:-2], -1).unsqueeze(-1)
        keys = torch.cat((linear, quadratic), dim=-1)
        queries = torch.cat((hkl * self.address[0], torch.ones_like(hkl[..., :1]) * self.address[1]), dim=-1)
        attn = (queries[:, None] @ keys.transpose(-2, -1)).softmax(-1)
        values = strength[:, :, None, None] * self.roots[m.remainder(4)]
        site_waves = attn @ values

        uniform = raw.new_zeros(MAX_SITES).softmax(0)
        wave = (uniform[None, None, None, :] @ site_waves.transpose(1, 2)).squeeze(-2) * self.pool_gain
        magnitude = F.relu(wave) + F.relu(-wave)
        squares = F.relu(magnitude.unsqueeze(-1) - self.knots) @ self.slopes
        intensity = squares.sum(-1)
        classes = torch.arange(CLASSES, device=tokens.device, dtype=raw.dtype)
        return -(intensity.unsqueeze(-1) - classes).abs()


def build_model():
    return DiffractionTransformer().eval()


def reference(record):
    """Evaluate the complex Fourier sum in double precision, then round the integer intensity."""
    encode(record)
    intensities = []
    for h, k, ell in record["reflections"]:
        total = sum(a * cmath.exp(0.5j * math.pi * (h*x + k*y + ell*z))
                    for x, y, z, a in record["sites"])
        power = abs(total) ** 2
        nearest = round(power)
        if abs(nearest - power) > 1e-8:
            raise ArithmeticError("Quarter-grid intensity must be an integer")
        intensities.append(nearest)
    return np.asarray(intensities, dtype=np.int64)


def generate(rng, size, index):
    """Draw centered, contrast-modified, diamond-type and general point motifs."""
    if not 4 <= size <= MAX_REFLECTIONS:
        raise ValueError("Study generator expects 4 through 32 reflections")
    family = index % 8
    if family in (0, 2):
        positions = np.array([[0, 0, 0], [2, 2, 2]])
    elif family in (1, 3):
        positions = np.array(FCC)
    elif family == 4:
        face = np.array(FCC)
        positions = np.concatenate((face, face + 1))
    else:
        count = int(rng.integers(1, MAX_SITES + 1))
        codes = rng.choice(64, size=count, replace=False)
        positions = np.stack(np.unravel_index(codes, (4, 4, 4)), axis=-1)
    strengths = rng.integers(1, 3, size=len(positions))
    if family in (0, 1, 4):
        strengths[:] = strengths[0]
    elif family in (2, 3):
        strengths[0], strengths[1] = 1, 2
    positions = (positions[:, rng.permutation(3)] + rng.integers(0, 4, size=3)) % 4
    sites = np.column_stack((positions, strengths))[rng.permutation(len(positions))]

    # A Friedel pair and a second harmonic exercise phase sign and order.
    base = np.zeros(3, dtype=int)
    while not np.any(base):
        base = rng.integers(-2, 3, size=3)
    refs = [base.tolist(), (-base).tolist(), (2*base).tolist()]
    while len(refs) < size:
        query = rng.integers(-MAX_INDEX, MAX_INDEX + 1, size=3).tolist()
        if any(query) and query not in refs:
            refs.append(query)
    return {"sites": sites.tolist(), "reflections": refs}


def example():
    return {"sites": [[*xyz, 1] for xyz in FCC],
            "reflections": [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]}


def format_output(record, prediction):
    encode(record)
    if len(prediction) != len(record["reflections"]):
        raise ValueError("Expected one intensity per requested reflection")
    return {"intensities": list(map(int, prediction))}
