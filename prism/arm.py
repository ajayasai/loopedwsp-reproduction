"""Solve discrete inverse kinematics for four links among lattice point obstacles."""
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_LINKS = 4
N_POSES = 3**N_LINKS
N_OBSTACLES = 8
LIMIT = 8
PAD = 16
TURNS = "LSR"


def encode(record):
    if not isinstance(record, dict) or set(record) != {"lengths", "obstacles", "targets"}:
        raise ValueError("Supply lengths, obstacles and targets")
    lengths = record["lengths"]
    if not isinstance(lengths, list) or len(lengths) != N_LINKS or any(type(n) is not int or n not in (1, 2) for n in lengths):
        raise ValueError("Supply four link lengths, each 1 or 2")
    for field, minimum, maximum in (("obstacles", 0, N_OBSTACLES), ("targets", 1, 64)):
        points = record[field]
        if not isinstance(points, list) or not minimum <= len(points) <= maximum:
            raise ValueError(f"Supply {minimum} through {maximum} {field}")
        if any(not isinstance(p, list) or len(p) != 2 or
               any(type(v) is not int or abs(v) > LIMIT for v in p) for p in points):
            raise ValueError(f"Each point in {field} needs integer x and y coordinates from -8 to 8")
    obstacles = record["obstacles"]
    if len({tuple(p) for p in obstacles}) != len(obstacles):
        raise ValueError("Use distinct obstacle positions")
    rows = [[n, 0] for n in lengths] + obstacles + [[PAD, PAD]] * (N_OBSTACLES - len(obstacles))
    return np.asarray(rows + record["targets"], dtype=np.int64)


class LatticeArmTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.directions = nn.Parameter(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(128.), requires_grad=False)

    def forward(self, tokens):
        raw = tokens.to(self.directions.dtype)
        lengths = raw[:, :N_LINKS, 0]
        obstacles = raw[:, N_LINKS:N_LINKS + N_OBSTACLES]
        targets = raw[:, N_LINKS + N_OBSTACLES:]
        device = raw.device
        pose_ids = torch.arange(N_POSES, device=device)
        place = 3 ** torch.arange(N_LINKS - 1, -1, -1, device=device)
        turns = 1 - (pose_ids[:, None] // place) % 3
        headings = turns.cumsum(-1).remainder(4).to(raw.dtype)
        dirs = torch.arange(4, device=device, dtype=raw.dtype)
        heading_weights = (-(headings[..., None] - dirs).square() * self.sharpness).softmax(-1)
        vectors = heading_weights @ self.directions

        # Each length-two link has two occupied lattice points beyond its start.
        # A length-one link has one occupied point and one inactive padding slot.
        steps = torch.stack((torch.ones_like(lengths), lengths - 1), dim=-1)
        moves = (vectors[None, :, :, None, :] * steps[:, None, :, :, None]).flatten(2, 3)
        points = F.pad(moves.cumsum(2), (0, 0, 1, 0))
        active = F.pad(steps.flatten(1).bool(), (1, 0), value=True)
        ids = points[..., 0] * (2 * LIMIT + 1) + points[..., 1]
        repeated = (ids[..., None] == ids[..., None, :]) & (active[:, None, :, None] & active[:, None, None, :])
        self_hit = repeated.tril(-1).any(dim=(-1, -2))

        # Nearest-obstacle attention reads a squared distance for each point.
        # Tied nearest obstacles carry the same distance, including exact contacts.
        distance = (points[:, :, :, None, :] - obstacles[:, None, None, :, :]).square().sum(-1)
        weights = (-self.sharpness * distance).softmax(-1)
        nearest = (weights * distance).sum(-1)
        blocked = ((nearest < 0.5) & active[:, None, :]).any(-1)
        reached = (targets[:, :, None, :] == points[:, None, :, -1, :]).all(-1)
        matches = reached & ~(blocked | self_hit)[:, None, :]

        # A final attention head selects the smallest admissible pose ID. The
        # always-eligible sentinel follows the 81 physical configurations.
        eligible = F.pad(matches.to(raw.dtype), (0, 1), value=1.)
        codes = torch.arange(N_POSES + 1, device=device, dtype=raw.dtype)
        scores = eligible * (N_POSES + 1) - codes
        first = (scores * self.sharpness).softmax(-1) @ codes
        answers = torch.stack((matches.sum(-1), first), dim=-1).flatten(1)
        return -(answers[..., None] - codes).abs()


def build_model():
    return LatticeArmTransformer().eval()


@lru_cache(maxsize=16)
def _walks(lengths):
    """Grow only simple lattice paths. A prefix that repeats a point is pruned."""
    result = []

    def extend(points, direction, code, link):
        if link == N_LINKS:
            result.append((code, points))
            return
        x, y = points[-1]
        dx, dy = direction
        for digit, (vx, vy) in enumerate(((-dy, dx), (dx, dy), (dy, -dx))):
            added = tuple((x + i*vx, y + i*vy) for i in range(1, lengths[link] + 1))
            if not any(point in points for point in added):
                extend(points + added, (vx, vy), 3*code + digit, link + 1)

    extend(((0, 0),), (1, 0), 0, 0)
    return tuple(result)


def _solutions(walks, obstacles):
    solutions = defaultdict(list)
    for code, points in walks:
        if obstacles.isdisjoint(points):
            solutions[points[-1]].append(code)
    return solutions


def reference(record):
    encode(record)
    solutions = _solutions(_walks(tuple(record["lengths"])), set(map(tuple, record["obstacles"])))
    labels = []
    for target in record["targets"]:
        feasible = solutions[tuple(target)]
        labels.extend((len(feasible), feasible[0] if feasible else N_POSES))
    return np.asarray(labels, dtype=np.int64)


def generate(rng, size, index):
    lengths = rng.integers(1, 3, size=4).tolist()
    walks = _walks(tuple(lengths))
    possible = sorted({point for _, points in walks for point in points} - {(0, 0)})
    n_obstacles = 0 if index % 5 == 0 else int(rng.integers(1, N_OBSTACLES + 1))
    obstacles = {possible[i] for i in rng.choice(len(possible), n_obstacles, replace=False)}
    solutions = _solutions(walks, obstacles)
    geometric = {points[-1] for _, points in walks}
    radius = sum(lengths)
    workspace = [(x,y) for x in range(-radius, radius+1) for y in range(-radius, radius+1) if abs(x)+abs(y) <= radius]
    groups = [sorted(p for p, ids in solutions.items() if len(ids) == 1),
              sorted(p for p, ids in solutions.items() if len(ids) > 1),
              sorted(geometric - set(solutions)),
              [p for p in workspace if p not in geometric]]
    targets = []
    for i in range(size):
        options = [p for p in groups[(i+index) % len(groups)] if p not in targets]
        if not options:
            options = [p for p in workspace if p not in targets]
        targets.append(options[int(rng.integers(len(options)))])
    return {"lengths": lengths, "obstacles": [list(p) for p in sorted(obstacles)],
            "targets": [list(p) for p in targets]}


def example():
    return {"lengths": [1,1,1,1], "obstacles": [[1,0]], "targets": [[2,2],[4,0],[0,0]]}


def format_output(record, prediction):
    if len(prediction) != 2*len(record["targets"]):
        raise ValueError("Expected a count and witness ID for each target")
    answers = []
    for target, count, code in zip(record["targets"], prediction[::2], prediction[1::2], strict=True):
        if not 0 <= count <= N_POSES or not 0 <= code <= N_POSES or (count == 0) != (code == N_POSES):
            raise ValueError("Invalid count or pose ID")
        witness = None if code == N_POSES else "".join(TURNS[(code // 3**i) % 3] for i in (3,2,1,0))
        answers.append({"target": target, "count": int(count), "first_pose": witness})
    return {"solutions": answers}
