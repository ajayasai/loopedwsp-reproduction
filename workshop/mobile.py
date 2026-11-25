"""Recover a missing integer counterweight for an ideal kinetic mobile.

The leaves are ornaments listed left to right. Each internal node is a
horizontal, massless rod whose two arm lengths are listed in breadth-first
order. A joint balances when the total hanging mass on each side times its
arm length is equal. Exactly one ornament is missing. Its allowed weight is
1 through 16 grams. Class zero means no weight in that range balances every
joint. This is an idealized arithmetic model for a tabletop design exercise.

For each candidate mass, a leaf starts with its own mass and zero imbalance.
Two softmax heads read the left and right child states inside each rod's fixed
two-token window. The block adds the child masses and carries the sum of the
absolute torque errors for every rod seen so far. The same seven stored
coefficients (six inside the block and one readout threshold) work at every
depth. The arm lengths, sibling grouping, candidate values, torque products
and ReLU activations are fixed inputs or operations. A separate integer
oracle solves the torque equations symbolically for the unknown mass.
"""

from math import gcd

import numpy as np


SIZES = (4, 8, 16, 32)
MAX_MASS = 32
MAX_COUNTERWEIGHT = 16
MAX_ARM = 512


def encode(record):
    if not isinstance(record, dict):
        raise ValueError("Expected an object with masses and arms")
    masses, arms = record.get("masses"), record.get("arms")
    if not isinstance(masses, list) or len(masses) not in SIZES:
        raise ValueError(f"Expected {SIZES} leaf masses, left to right")
    if sum(x is None for x in masses) != 1 or any(
        x is not None and (type(x) is not int or not 1 <= x <= MAX_MASS) for x in masses
    ):
        raise ValueError("Supply one null mass and integers from 1 through 32 for the other ornaments")
    n = len(masses)
    if not isinstance(arms, list) or len(arms) != n - 1:
        raise ValueError(f"Supply {n - 1} pairs of arm lengths in breadth-first order")
    for a in arms:
        if not isinstance(a, list) or len(a) != 2 or any(type(x) is not int or not 1 <= x <= MAX_ARM for x in a):
            raise ValueError("Each rod needs two integer arm lengths from 1 through 512")
    rows = np.zeros((2*n - 1, 3), dtype=np.int64)
    rows[:n-1, :2] = arms
    rows[n-1:, 2] = [0 if mass is None else mass for mass in masses]
    return rows


def build_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class BalanceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Scalar query for each head. Child keys are fixed side
            # coordinates (-1,+1), yielding a score gap of 160 in FP32.
            self.queries = nn.Parameter(torch.tensor([-80., 80.]), requires_grad=False)
            self.torque_sign = nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)
            self.merge = nn.Parameter(torch.tensor([1., 1.]), requires_grad=False)

        def forward(self, state, arms):
            batch, children, candidates, fields = state.shape
            rods = children // 2
            keys = self.queries.new_tensor([-1., 1.])
            attention = (self.queries[:, None] * keys[None, :]).softmax(-1)
            values = state.reshape(batch, rods, 2, candidates*fields)
            gathered = (attention @ values).reshape(batch, rods, 2, candidates, fields)
            masses, errors = gathered.unbind(-1)
            # Each row now contains a pair of subtree masses and one error
            # per side, for every possible counterweight. Arm lengths are
            # direct physical inputs to the signed torque product.
            signed_arms = arms.to(state.dtype) * self.torque_sign
            delta = (masses * signed_arms[:, :, :, None]).sum(2)
            new_error = F.relu(delta) + F.relu(-delta)
            mass = masses.transpose(-1, -2) @ self.merge
            error = errors.transpose(-1, -2) @ self.merge + new_error
            return torch.stack((mass, error), -1)

    class MobileTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = BalanceBlock()
            self.impossible_score = nn.Parameter(torch.tensor(-0.5), requires_grad=False)

        def forward(self, tokens):
            n = (tokens.shape[1] + 1) // 2
            if n not in SIZES or tokens.shape[1:] != (2*n - 1, 3):
                raise ValueError("Expected a complete tree of rod and ornament tokens")
            mass = tokens[:, n-1:, 2:3]
            candidate = torch.arange(1, MAX_COUNTERWEIGHT + 1, device=tokens.device)
            mass = torch.where(mass == 0, candidate, mass).to(self.impossible_score.dtype)
            state = torch.stack((mass, torch.zeros_like(mass)), -1)
            width = n
            while width > 1:
                parents = width // 2
                arms = tokens[:, parents-1:width-1, :2]
                state = self.block(state, arms)
                width = parents
            scores = -state[..., 1]
            # All torque errors are integers. A valid candidate scores zero
            # and each invalid candidate scores at most minus one.
            fallback = self.impossible_score.expand(tokens.shape[0], 1, 1)
            return torch.cat((fallback, scores), -1)

    return MobileTransformer().eval()


def reference(record):
    """Solve a system of linear integer equations; the tree is visited once."""
    encode(record)
    masses, arms = record["masses"], record["arms"]
    n = len(masses)
    equations = []

    def subtree(position):
        if position >= n - 1:
            weight = masses[position - (n - 1)]
            return (1, 0) if weight is None else (0, weight)
        left_slope, left_base = subtree(2*position + 1)
        right_slope, right_base = subtree(2*position + 2)
        left_arm, right_arm = arms[position]
        equations.append((left_arm*left_slope - right_arm*right_slope,
                          right_arm*right_base - left_arm*left_base))
        return left_slope + right_slope, left_base + right_base

    subtree(0)
    pivot, value = next((slope, rhs) for slope, rhs in equations if slope)
    weight, remainder = divmod(value, pivot)
    consistent = (remainder == 0 and 1 <= weight <= MAX_COUNTERWEIGHT
                  and all(slope*weight == rhs for slope, rhs in equations))
    return np.array([weight if consistent else 0], dtype=np.int64)


def example():
    return {"masses": [2, None, 3, 5], "arms": [[4, 3], [2, 1], [5, 3]]}


def generate(rng, size, index):
    """Mix balanced designs with out-of-range, remote, root and fractional failures."""
    if size not in SIZES:
        raise ValueError(f"Expected one of {SIZES} ornaments")
    variant = index % 8
    missing = int(rng.integers(size))
    masses = rng.integers(1, MAX_MASS + 1, size=size).tolist()
    masses[missing] = 1 + (index // 2) % MAX_COUNTERWEIGHT
    if variant == 1:
        masses[missing] = int(rng.integers(17, 33))
    elif variant == 7:
        # A 2:1 torque equation with an odd known neighbor needs a half gram.
        masses[missing ^ 1] = 2*int(rng.integers(16)) + 1

    combined = [0]*(size-1) + masses
    arms = [[0, 0] for _ in range(size-1)]
    for i in range(size - 2, -1, -1):
        left, right = combined[2*i+1:2*i+3]
        divisor = gcd(left, right)
        left_arm, right_arm = right // divisor, left // divisor
        factor = int(rng.integers(1, 1 + min(4, MAX_ARM // max(left_arm, right_arm))))
        arms[i] = [factor*left_arm, factor*right_arm]
        combined[i] = left + right

    nearest = size // 2 - 1 + missing // 2
    if variant in (3, 5):
        # Disturb a fully known bottom rod, or make the root contradict the
        # locally valid counterweight at the missing ornament's own rod.
        remote = [i for i in range(size // 2 - 1, size - 1) if i != nearest]
        rod = int(rng.choice(remote)) if variant == 3 else 0
        end = int(rng.integers(2))
        arms[rod][end] += 1 if arms[rod][end] < MAX_ARM else -1
    elif variant == 7:
        arms[nearest][missing % 2] = 2
        arms[nearest][1 - missing % 2] = 1

    masses[missing] = None
    return {"masses": masses, "arms": arms}


def format_output(record, prediction):
    if len(prediction) != 1 or not 0 <= int(prediction[0]) <= MAX_COUNTERWEIGHT:
        raise ValueError("Expected one result in the range 0 through 16")
    value = int(prediction[0])
    return {"counterweight_grams": value or None}
