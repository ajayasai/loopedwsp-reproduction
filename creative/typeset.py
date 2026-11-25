"""A weight-programmed paragraph compositor for monospaced text.

Inputs contain 1–32 words with one space between adjacent words on a line
and a width of 1–32 columns. Every word must fit. The objective is the sum
of squared unused columns on all lines except the final line. Among equal
scores, take the longer first line, then the longer second line, and so on.
Output token j marks whether a line ends after word j.

A token represents a word boundary. The shared attention block chooses a
feasible line end, using its geometric line cost plus the other token's
current suffix cost. Its values carry the suffix's break flags, word-column
coordinate, terminal bit and cost. A ReLU feedforward layer adds the fitted
line's integer squared slack. After t rounds, boundaries within t words of
the end have exact solutions. The first token therefore knows every break
after n rounds. Boundary coordinates and the line-fit mask come from input
word lengths alone. All calculations in forward are batched device operations.
"""
import string

import numpy as np


MAX_WIDTH = 32
MAX_WORDS = 32


def encode(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a paragraph object with width and words")
    width, words = record.get("width"), record.get("words")
    if type(width) is not int or not 1 <= width <= MAX_WIDTH:
        raise ValueError(f"Page width must be an integer from 1 through {MAX_WIDTH}")
    if not isinstance(words, list) or not 1 <= len(words) <= MAX_WORDS:
        raise ValueError(f"Provide 1 through {MAX_WORDS} words")
    for word in words:
        if (not isinstance(word, str) or not word or not word.isascii() or not word.isprintable()
                or any(char.isspace() for char in word) or len(word) > width):
            raise ValueError("Each word must contain printable ASCII without spaces and fit the given width")
    return np.asarray([(len(word), width) for word in words], dtype=np.int64)


def build_model():
    """Build the frozen, tied transformer. The oracle and generator need only NumPy."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LineBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # A higher score chooses smaller total cost and then the farther endpoint.
            # The minimum winner/runner-up gap of 128 prevents softmax mixture in FP32.
            self.address = nn.Parameter(torch.tensor([-128., 128.]), requires_grad=False)
            # For integer s, relu(s) + 2*sum_{t=1}^{W-1} relu(s-t) equals s squared.
            self.threshold = nn.Parameter(-torch.arange(MAX_WIDTH, dtype=torch.float32), requires_grad=False)
            coefficients = torch.full((MAX_WIDTH,), 2.)
            coefficients[0] = 1.
            self.spline = nn.Parameter(coefficients, requires_grad=False)

        def forward(self, suffix, breaks, geometry):
            prefix, boundary_ids, nonterminal, one_hot, line_bias, max_column = geometry
            m = prefix.shape[1]
            keys = self.address[0] * m * suffix + self.address[1] * boundary_ids
            # The scalar query of every boundary is one. A relative-position bias
            # expresses the input line's squared slack and masks only illegal spans.
            attention = (keys[:, None, :] + line_bias).softmax(dim=-1)
            values = torch.cat((suffix.unsqueeze(-1), prefix.unsqueeze(-1),
                                (1 - nonterminal).unsqueeze(-1), breaks + one_hot), dim=-1)
            gathered = attention @ values
            # A selected final endpoint has a free line. The terminal token
            # attends to itself and is reset to the empty zero-cost suffix.
            slack = max_column - gathered[:, :, 1] - (MAX_WIDTH + 1) * gathered[:, :, 2]
            line_cost = F.relu(slack.unsqueeze(-1) + self.threshold) @ self.spline
            return ((gathered[:, :, 0] + line_cost) * nonterminal,
                    gathered[:, :, 3:] * nonterminal.unsqueeze(-1))

    class Compositor(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = LineBlock()

        def forward(self, tokens):
            if tokens.ndim != 3 or tokens.shape[-1] != 2 or not 1 <= tokens.shape[1] <= MAX_WORDS:
                raise ValueError("Expected [batch,1..32,2] length/width tokens")
            batch, n = tokens.shape[:2]
            dtype = self.block.address.dtype
            prefix = F.pad((tokens[:, :, 0] + 1).cumsum(dim=1), (1, 0)).to(dtype)
            ids = torch.arange(n + 1, device=tokens.device)
            nonterminal = (ids < n).to(dtype).expand(batch, -1)
            one_hot = torch.eye(n + 1, device=tokens.device, dtype=dtype).expand(batch, -1, -1)
            max_column = prefix + tokens[:, :1, 1] + 1
            unused = max_column[:, :, None] - prefix[:, None, :]
            forward = ids[:, None] < ids[None, :]
            final_self = (ids[:, None] == n) & (ids[None, :] == n)
            allowed = (forward & (unused >= 0)) | final_self
            edge_cost = unused.square() * nonterminal[:, None, :]
            line_bias = (self.block.address[0] * (n + 1) * edge_cost).masked_fill(~allowed, -torch.inf)
            geometry = prefix, ids, nonterminal, one_hot, line_bias, max_column
            suffix = torch.zeros_like(prefix)
            breaks = torch.zeros_like(one_hot)
            for _ in range(n):
                suffix, breaks = self.block(suffix, breaks, geometry)
            selected = breaks[:, 0, 1:]
            return torch.stack((1 - selected, selected), dim=-1)

    return Compositor().eval()


def reference(record):
    """Solve the exact layout with backward integer DP and follow the best endpoints."""
    lengths = encode(record)[:, 0].tolist()
    width, n = record["width"], len(lengths)
    best = [0] * (n + 1)
    following = [n] * n
    for start in range(n - 1, -1, -1):
        alternatives = []
        occupied = -1
        for stop in range(start + 1, n + 1):
            occupied += 1 + lengths[stop - 1]
            if occupied > width:
                break
            penalty = (width - occupied) ** 2 if stop != n else 0
            alternatives.append((penalty + best[stop], -stop))
        best[start], negative_end = min(alternatives)
        following[start] = -negative_end
    answer = np.zeros(n, dtype=np.int64)
    cursor = 0
    while cursor < n:
        cursor = following[cursor]
        answer[cursor - 1] = 1
    return answer


def example():
    # Greedy wrapping gives "Ink on" / "my" / "plate", with raggedness 16.
    return {"width": 6, "words": ["Ink", "on", "my", "plate"]}


def generate(rng, size, index):
    """Draw synthetic single-width words, sometimes planting a greedy or tie case."""
    if not 1 <= size <= MAX_WORDS:
        raise ValueError("Unsupported paragraph length")
    width = int(rng.integers(4, MAX_WIDTH + 1))
    lengths = rng.integers(1, width + 1, size=size).tolist()
    # Scaled motifs preserve fit and the relative order of every layout cost.
    # A full-width fifth word forces the motif's final line to end at word four.
    if size >= 4 and index % 8 in (0, 1):
        tie = index % 8 == 1
        unit = int(rng.integers(1, 5))
        base_width, motif = (5, [1, 2, 1, 4]) if tie else (6, [3, 2, 2, 5])
        width = (base_width + 1) * unit - 1
        lengths = [(length + 1) * unit - 1 for length in motif]
        if size > 4:
            lengths += [width] + rng.integers(1, width + 1, size=size - 5).tolist()
    elif index % 8 in (2, 3):
        lengths = rng.integers(1, 1 + max(1, width // 3), size=size).tolist()
    alphabet = string.ascii_lowercase
    words = ["".join(alphabet[int(x)] for x in rng.integers(len(alphabet), size=k)) for k in lengths]
    return {"width": width, "words": words}


def format_output(record, prediction):
    tokens = encode(record)
    flags = np.asarray(prediction)
    if flags.shape != (len(tokens),) or not np.isin(flags, [0, 1]).all() or not flags[-1]:
        raise ValueError("Expected one binary break flag per word and a break after the final word")
    lines, start = [], 0
    for stop in np.flatnonzero(flags) + 1:
        lines.append(" ".join(record["words"][start:stop]))
        start = stop
    return {"lines": lines, "raggedness": sum((record["width"] - len(line))**2 for line in lines[:-1])}
