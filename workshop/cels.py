"""Composite up to 32 rectangular animation cels over an opaque 8×8 RGB4 background.

Rectangles use half-open [x0, y0, x1, y1] bounds measured from the top left.
The cels array is ordered from bottom to top. Opacity is an integer in units
of one half. All channels are 0..15 and half-integer blends round upward at
each layer. These rules define a discrete compositor for pixel animation.

One tied transformer block is applied per cel. Each canvas pixel attends to
the current cel, with an input geometry mask and a clear fallback token.
The value contains the cel color and the three opacity modes. A ReLU spline
computes the rounded half blend and ReLU gates select the new pixel color.
The score scale ensures an addressing gap of at least 128 in FP32. The
initial background and final class coordinates are fixed input/output maps.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SIDE = 8
LEVELS = 16
MAX_CELS = 32


def _color(rgb):
    if (not isinstance(rgb, list) or len(rgb) != 3 or
            any(type(v) is not int or not 0 <= v < LEVELS for v in rgb)):
        raise ValueError("Expected a three-channel RGB list with integers from 0 through 15")
    return rgb


def encode(record):
    if not isinstance(record, dict):
        raise ValueError("Expected an animation object with background and cels")
    bg = _color(record.get("background"))
    cels = record.get("cels")
    if not isinstance(cels, list) or len(cels) > MAX_CELS:
        raise ValueError("Supply a bottom-to-top list of at most 32 cels")
    tokens = [[0, 0, SIDE, SIDE, *bg, 2]]
    for cel in cels:
        if not isinstance(cel, dict):
            raise ValueError("Each cel must contain rect, rgb and opacity")
        rect = cel.get("rect")
        if (not isinstance(rect, list) or len(rect) != 4 or any(type(v) is not int for v in rect)
                or not (0 <= rect[0] < rect[2] <= SIDE and 0 <= rect[1] < rect[3] <= SIDE)):
            raise ValueError("Expected nonempty [x0,y0,x1,y1] bounds within the 8×8 canvas")
        opacity = cel.get("opacity")
        if type(opacity) is not int or opacity not in (0, 1, 2):
            raise ValueError("Cel opacity must be 0, 1 or 2 half-units")
        tokens.append([*rect, *_color(cel.get("rgb")), opacity])
    return np.asarray(tokens, dtype=np.int64)


class CompositeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.address = nn.Parameter(torch.tensor([512., -256., 256.]), requires_grad=False)
        # For integer x in [0,30], this alternating hinge sum is ceil(x / 2).
        self.bias = nn.Parameter(-torch.arange(2 * (LEVELS - 1), dtype=torch.float32), requires_grad=False)
        signs = torch.tensor([1., -1.]).repeat(LEVELS - 1)
        self.slope = nn.Parameter(signs, requires_grad=False)
        self.gate_margin = nn.Parameter(torch.tensor(float(LEVELS)), requires_grad=False)

    def forward(self, canvas, query, keys, coverage, values):
        scores = (query * self.address) @ keys.T
        attention = scores.masked_fill(~coverage, -torch.inf).softmax(-1)
        selected = attention @ values
        rgb, modes = selected.split(3, dim=-1)
        rounded_half = F.relu((canvas + rgb).unsqueeze(-1) + self.bias) @ self.slope
        options = torch.stack((canvas, rounded_half, rgb), dim=-1)
        return F.relu(options + (modes.unsqueeze(-2) - 1) * self.gate_margin).sum(-1)


class CelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = CompositeBlock()

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 8 or not 1 <= tokens.shape[1] <= MAX_CELS + 1:
            raise ValueError("Expected [batch, 1+cel_count, 8] tokens, starting with the background")
        dtype = self.block.address.dtype
        canvas = tokens[:, :1, 4:7].to(dtype).expand(-1, SIDE * SIDE, -1)
        cels = tokens[:, 1:]
        count = cels.shape[1]
        position = torch.arange(count, dtype=dtype, device=tokens.device)
        # Real-key logits are 256*(2*s*j-j²). The clear token's score lies
        # halfway between the current layer and the nearest different layer.
        keys = torch.stack((position, position.square(), torch.zeros_like(position)), dim=-1)
        keys = torch.cat((keys, keys.new_tensor([[0, 0, 1]])), dim=0)
        queries = torch.stack((position, torch.ones_like(position), position.square() - 0.5), dim=-1)

        pixel = torch.arange(SIDE * SIDE, device=tokens.device)
        x, y = (pixel % SIDE)[:, None], (pixel // SIDE)[:, None]
        x0, y0, x1, y1 = cels[:, None, :, :4].unbind(dim=-1)
        coverage = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        coverage = F.pad(coverage, (0, 1), value=True)
        modes = F.one_hot(cels[:, :, 7], num_classes=3)
        values = torch.cat((cels[:, :, 4:7], modes), dim=-1).to(dtype)
        clear = values.new_tensor([0, 0, 0, 1, 0, 0]).expand(tokens.shape[0], 1, -1)
        values = torch.cat((values, clear), dim=1)
        for query in queries.unbind():
            canvas = self.block(canvas, query, keys, coverage, values)
        classes = torch.arange(LEVELS, dtype=dtype, device=tokens.device)
        return -(canvas.reshape(tokens.shape[0], -1, 1) - classes).abs()


def build_model():
    return CelTransformer().eval()


def reference(record):
    """Rasterize each cel into the affected NumPy slice using integer arithmetic."""
    encode(record)
    image = np.broadcast_to(np.asarray(record["background"], dtype=np.int64), (SIDE, SIDE, 3)).copy()
    for cel in record["cels"]:
        x0, y0, x1, y1 = cel["rect"]
        region = image[y0:y1, x0:x1]
        alpha = cel["opacity"]
        region[:] = (alpha * np.asarray(cel["rgb"]) + (2 - alpha) * region + 1) // 2
    return image.ravel()


def _random_rect(rng):
    x0, y0 = rng.integers(SIDE, size=2).tolist()
    return [x0, y0, int(rng.integers(x0 + 1, SIDE + 1)), int(rng.integers(y0 + 1, SIDE + 1))]


def generate(rng, size, index):
    if not 0 <= size <= MAX_CELS:
        raise ValueError("Unsupported number of cels")
    bg = rng.integers(LEVELS, size=3).tolist()
    cels = [{"rect": _random_rect(rng), "rgb": rng.integers(LEVELS, size=3).tolist(),
             "opacity": int(rng.choice([0, 1, 1, 2]))} for _ in range(size)]
    if size >= 2:
        # Paired inset cels, half-tints over the entire frame, opaque
        # occlusion and a final clear pass supplement free-form stacks.
        motif = index % 5
        if motif == 0:
            cels[-2].update(rect=[1, 1, 7, 7], opacity=1)
            cels[-1].update(rect=[3, 2, 8, 6], opacity=1)
        elif motif == 1:
            for cel in cels[-2:]:
                cel.update(rect=[0, 0, SIDE, SIDE], opacity=1)
        elif motif == 2:
            cels[-2].update(rect=cels[-1]["rect"].copy(), opacity=1)
            cels[-1]["opacity"] = 2
        elif motif == 3:
            cels[-1].update(rect=[0, 0, SIDE, SIDE], opacity=0)
    return {"background": bg, "cels": cels}


def example():
    return {"background": [1, 2, 5], "cels": [
        {"rect": [2, 1, 6, 7], "rgb": [14, 6, 1], "opacity": 2},
        {"rect": [3, 2, 5, 6], "rgb": [15, 13, 4], "opacity": 2},
        {"rect": [0, 3, 8, 5], "rgb": [1, 7, 15], "opacity": 1},
    ]}


def format_output(record, prediction):
    encode(record)
    channels = np.asarray(prediction)
    if (channels.shape != (SIDE * SIDE * 3,) or channels.dtype.kind not in "iu"
            or np.any(channels < 0) or np.any(channels >= LEVELS)):
        raise ValueError("Expected 192 RGB4 channel IDs for the 8×8 frame")
    return {"pixels": channels.reshape(SIDE, SIDE, 3).tolist()}
