"""Decode a bounded uncontracted UEB alphabet and numeric subset from six-dot cells."""

import string

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Physical raised dots for the English alphabet, in a–z order.
LETTERS = ("1", "12", "14", "145", "15", "124", "1245", "125", "24", "245",
           "13", "123", "134", "1345", "135", "1234", "12345", "1235", "234", "2345",
           "136", "1236", "2456", "1346", "13456", "1356")
NUMBER, CAPITAL, GRADE1 = "3456", "6", "56"
GLYPHS = ("", *LETTERS, NUMBER, CAPITAL, GRADE1)
PRINT = ("", " ", *string.ascii_lowercase, *string.ascii_uppercase, *string.digits)
MAX_CELLS = 128


def encode(record):
    if not isinstance(record, dict) or set(record) != {"cells"}:
        raise ValueError("Supply a cells list")
    cells = record["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= MAX_CELLS:
        raise ValueError("Use 1 to 128 cells")
    if any(not isinstance(cell, str) or cell not in GLYPHS for cell in cells):
        raise ValueError("Use a supported six-dot pattern, with raised dots in ascending order")
    # Local syntax checks prevent dangling and doubled indicators. Only raw dot
    # occupancy is passed onward. The network determines the reading mode.
    for i, cell in enumerate(cells):
        if cell in (NUMBER, CAPITAL, GRADE1):
            permitted = LETTERS if cell == CAPITAL else LETTERS[:10]
            if i+1 == len(cells) or cells[i+1] not in permitted:
                raise ValueError("An indicator must precede an appropriate letter cell")
    return np.asarray([[int(str(dot) in cell) for dot in range(1,7)] for cell in cells], dtype=np.int64)


class BrailleReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.dot_weights = nn.Parameter(torch.tensor([1., 2., 4., 8., 16., 32.]), requires_grad=False)
        masks = [sum(1 << (int(dot)-1) for dot in cell) for cell in GLYPHS]
        self.glyph_keys = nn.Parameter(torch.tensor(masks, dtype=torch.float32), requires_grad=False)
        # Each glyph offers three readings: ordinary text, initial capital, number.
        readings = [[1,1,1]]
        for i in range(26):
            readings.append([2+i, 28+i, 54+(i+1)%10 if i < 10 else 2+i])
        readings.extend([[0,0,0]] * 3)
        self.readings = nn.Parameter(torch.tensor(readings, dtype=torch.float32), requires_grad=False)
        self.gain = nn.Parameter(torch.tensor(64.), requires_grad=False)

    def forward(self, tokens):
        raw = tokens.to(self.dot_weights.dtype)
        masks = raw @ self.dot_weights
        glyph_attention = (-self.gain * (masks[...,None]-self.glyph_keys).square()).softmax(-1)
        readings = glyph_attention @ self.readings

        # Within the restricted alphabet only the a–j patterns allow numeric
        # mode to persist. Attend to the closest earlier cell that sets or clears
        # the mode. An initial blank supplies a key for the start of each line.
        permits_digit = glyph_attention[..., 1:11].sum(-1) > .5
        resets_or_starts = F.pad(~permits_digit, (1,0), value=True)
        number_sign = F.pad(glyph_attention[..., 27], (1,0))
        positions = torch.arange(tokens.size(1)+1, device=tokens.device)
        previous = positions[None,:] < positions[1:,None]
        allowed = previous[None] & resets_or_starts[:,None,:]
        scores = self.gain * positions.to(raw.dtype)
        attention = scores.expand_as(allowed).masked_fill(~allowed, -torch.inf).softmax(-1)
        numeric = (attention @ number_sign[...,None]).squeeze(-1) > .5
        after_capital = F.pad(glyph_attention[..., 28], (1,0))[:, :-1] > .5
        symbol = torch.where(after_capital, readings[...,1], readings[...,0])
        symbol = torch.where(numeric & permits_digit, readings[...,2], symbol)
        labels = torch.arange(len(PRINT), dtype=raw.dtype, device=tokens.device)
        return -(symbol[...,None]-labels).abs()


def build_model():
    return BrailleReader().eval()


def reference(record):
    """Read left to right with explicit number and capital state."""
    encode(record)
    numeric = uppercase = False
    labels = []
    for cell in record["cells"]:
        if cell in (NUMBER, CAPITAL, GRADE1):
            numeric, uppercase = cell == NUMBER, cell == CAPITAL
            labels.append(0)
        elif cell == "":
            numeric = uppercase = False
            labels.append(1)
        else:
            letter = LETTERS.index(cell)
            if numeric and letter < 10:
                labels.append(54+(letter+1)%10)
            else:
                numeric = False
                labels.append((28 if uppercase else 2)+letter)
            uppercase = False
    return np.asarray(labels, dtype=np.int64)


def generate(rng, size, index):
    if not 8 <= size <= MAX_CELLS:
        raise ValueError("Benchmark lines have 8 to 128 cells")
    def draw(alphabet, count):
        return rng.choice(alphabet, size=count).tolist()
    def digits(count):
        return draw(LETTERS[:10], count)

    if index % 5 == 0:
        cells = [NUMBER] + digits(size-1)
    elif index % 5 == 1:
        cut = int(rng.integers(2, size-2))
        cells = [NUMBER] + digits(cut-1) + [GRADE1] + digits(1) + draw(LETTERS, size-cut-2)
    elif index % 5 == 2:
        cut = int(rng.integers(2, size-2))
        cells = [NUMBER] + digits(cut-1) + draw(LETTERS[10:], 1) + draw(LETTERS, size-cut-1)
    elif index % 5 == 3:
        cut = int(rng.integers(2, size-3))
        cells = [NUMBER] + digits(cut-1) + ["", CAPITAL] + draw(LETTERS, size-cut-2)
    else:
        cut = int(rng.integers(3, size-2))
        cells = [CAPITAL] + draw(LETTERS, cut-1) + ["", NUMBER] + digits(size-cut-2)
    return {"cells": cells}


def example():
    # Room 204b
    return {"cells": [CAPITAL, "1235", "135", "135", "134", "", NUMBER, "12", "245", "145", GRADE1, "12"]}


def format_output(record, prediction):
    if len(prediction) != len(record["cells"]) or any(not 0 <= i < len(PRINT) for i in prediction):
        raise ValueError("Expected one print label per input cell")
    return {"text": "".join(PRINT[i] for i in prediction)}
