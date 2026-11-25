"""Preview the visible eight-color mosaic on a folded, double-sided sheet.

The JSON object supplies front and back grids in the same coordinates, viewed
through the original front-up sheet. Each side length is a power of two from
1 to 32. A crease instruction names the moving half, its motion (over or
under) and the stationary half, e.g. "left_over_right" or "top_under_bottom".
Each center fold halves the corresponding dimension. The folded sheet keeps
its screen orientation and is cropped to its new bounding box.

The tied block uses two softmax attention heads that address the moving and
stationary input stacks for each output cell. Positional features use the
crease geometry and are independent of the colors. A shared surface
projection and ReLU gate reverse the moving stack's face orientation and
route the new outer colors. The independent oracle retains every paper layer
and physically flips subgrids and stacks. Neither implementation uses the
other's fold update.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


COLORS = 8
# Bit 2 selects the vertical axis, bit 1 moves the high-coordinate half,
# and bit 0 places that half over the stationary half.
FOLDS = tuple(
    f"{moving}_{motion}_{stationary}"
    for moving, stationary in (("left", "right"), ("right", "left"),
                               ("top", "bottom"), ("bottom", "top"))
    for motion in ("under", "over")
)


def _sheet(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a front grid, a back grid and a list of folds")
    grids = []
    for face in ("front", "back"):
        grid = np.asarray(record.get(face))
        if (grid.ndim != 2 or grid.dtype.kind not in "iu"
                or any(n not in (1, 2, 4, 8, 16, 32) for n in grid.shape)
                or np.any(grid < 0) or np.any(grid >= COLORS)):
            raise ValueError(f"{face} must be a power-of-two color grid with integers from 0 to 7")
        grids.append(grid)
    if grids[0].shape != grids[1].shape:
        raise ValueError("The two faces must have the same shape")
    folds = record.get("folds")
    if not isinstance(folds, list) or any(fold not in FOLDS for fold in folds):
        raise ValueError(f"folds must be a list of instructions from {FOLDS}")
    height, width = grids[0].shape
    for direction in folds:
        if direction.startswith(("left", "right")):
            width //= 2
        else:
            height //= 2
        if min(height, width) < 1:
            raise ValueError("A crease must split two equal, nonempty halves")
    return grids[0], grids[1], folds, (height, width)


def encode(record):
    """Serialize raw colors, original width and instructions. No fold is evaluated."""
    front, back, folds, _ = _sheet(record)
    tokens = np.zeros((front.size, 3 + len(folds)), dtype=np.int64)
    tokens[:, 0] = front.ravel()
    tokens[:, 1] = back.ravel()
    tokens[0, 2] = front.shape[1]
    tokens[0, 3:] = [FOLDS.index(fold) for fold in folds]
    return tokens


class FoldBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.diag(torch.tensor([2., 1.])), requires_grad=False)
        self.key = nn.Parameter(torch.diag(torch.tensor([16., -16.])), requires_grad=False)
        # Rows are stationary top/bottom, moving top/bottom, respectively.
        # Columns are under-fold top/bottom and over-fold top/bottom.
        routes = torch.tensor([[1., 0., 0., 0.],
                               [0., 0., 0., 1.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.]])
        self.surface_routes = nn.Parameter(routes, requires_grad=False)

    def forward(self, state, width, command):
        batch, area, _, colors = state.shape
        vertical = (command & 4) != 0
        move_high = (command & 2) != 0
        over = (command & 1).to(state.dtype)
        new_width = torch.where(vertical, width, width // 2)
        new_area = area // 2
        # Local coordinates are measured on the cropped result. Express the two
        # candidate stack locations on the old canvas, before looking at colors.
        out = torch.arange(new_area, device=state.device)[None, :]
        r = out // new_width
        c = out % new_width
        h = area // width
        sr = r + (vertical & ~move_high) * (h // 2)
        sc = c + (~vertical & ~move_high) * (width // 2)
        mr = torch.where(vertical, h - 1 - sr, sr)
        mc = torch.where(vertical, sc, width - 1 - sc)
        addresses = torch.stack((sr * width + sc, mr * width + mc), dim=1).to(state.dtype)
        j = torch.arange(area, device=state.device, dtype=state.dtype)
        # QK gives 16*(2*i*j-j*j). Its row softmax equals that of -16*(i-j)^2.
        q = torch.stack((addresses, torch.ones_like(addresses)), dim=-1) @ self.query
        k = torch.stack((j, j.square()), dim=-1) @ self.key
        weights = (q @ k.T).softmax(dim=-1)
        faces = weights @ state.reshape(batch, 1, area, 2 * colors)
        faces = faces.transpose(1, 2).reshape(batch, new_area, 4, colors)
        routes = faces.transpose(-1, -2) @ self.surface_routes
        under_faces, over_faces = routes.split(2, dim=-1)
        gate = over[:, :, None, None]
        next_faces = F.relu(under_faces - gate) + F.relu(over_faces - (1 - gate))
        return next_faces.transpose(-1, -2), new_width


class PaperFoldTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = FoldBlock()
        self.visible = nn.Parameter(torch.tensor([1., 0.]), requires_grad=False)

    def forward(self, tokens):
        state = F.one_hot(tokens[:, :, :2], num_classes=COLORS).to(self.visible.dtype)
        width = tokens[:, 0, 2:3]
        for step in range(3, tokens.shape[2]):
            state, width = self.block(state, width, tokens[:, 0, step:step+1])
        return state.transpose(-1, -2) @ self.visible


def build_model():
    return PaperFoldTransformer().eval()


def reference(record):
    front, back, folds, _ = _sheet(record)
    # Each stack lists layers from bottom to top. Each layer stores its two
    # currently oriented faces, even when many layers hide it from view.
    grid = np.empty(front.shape, dtype=object)
    for i, j in np.ndindex(grid.shape):
        grid[i, j] = [(int(front[i, j]), int(back[i, j]))]
    for fold in folds:
        moving, motion, _ = fold.split("_")
        axis = 1 if moving in ("left", "right") else 0
        low, high = np.split(grid, 2, axis=axis)
        flap, base = (low, high) if moving in ("left", "top") else (high, low)
        flap = np.flip(flap, axis=axis)
        grid = np.empty(base.shape, dtype=object)
        for i, j in np.ndindex(grid.shape):
            turned = [(bottom, top) for top, bottom in reversed(flap[i, j])]
            grid[i, j] = base[i, j] + turned if motion == "over" else turned + base[i, j]
    return np.array([stack[-1][0] for stack in grid.flat], dtype=np.int64)


def generate(rng, size, index):
    """Produce independent color fields or paired geometric motifs and legal folds."""
    rounds = {4: 1, 8: 1, 16: 2, 32: 3}
    if size not in rounds:
        raise ValueError(f"Benchmark sizes are {tuple(rounds)}")
    rows, cols = np.indices((size, size))
    faces = []
    for face in range(2):
        if index % 3 == 0:
            colors = rng.integers(COLORS, size=(size, size))
        else:
            tile = int(rng.integers(1, 4))
            x = (cols + int(rng.integers(COLORS))) // tile
            y = (rows + int(rng.integers(COLORS))) // tile
            pattern = x + 3*y if (index + face) % 2 == 0 else x ^ y
            colors = rng.permutation(COLORS)[pattern % COLORS]
        faces.append(colors.tolist())
    axes = rng.permutation([0, 1] * rounds[size])
    folds = [FOLDS[4*int(axis) + int(rng.integers(4))] for axis in axes]
    return {"front": faces[0], "back": faces[1], "folds": folds}


def example():
    return {"front": [[0, 1, 2, 3], [4, 5, 6, 7], [2, 3, 4, 5], [6, 7, 0, 1]],
            "back": [[7, 7, 6, 6], [5, 5, 4, 4], [3, 3, 2, 2], [1, 1, 0, 0]],
            "folds": ["left_under_right", "top_over_bottom"]}


def format_output(record, prediction):
    *_, shape = _sheet(record)
    return {"visible": np.asarray(prediction).reshape(shape).tolist()}
