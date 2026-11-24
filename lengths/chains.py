"""Shuffled integer assignment chains, with a single queried variable."""

import numpy as np


SIZES = (1, 4, 8, 12, 16, 24, 32, 63)
DEV_SIZES = (1, 4, 8)
COUNT = 64
BASE, PLUS, MINUS, QUERY = range(4)
MISSING = 64


def reference(rows):
    """Evaluate the displayed definitions by following names from the query."""
    definitions = {int(lhs): (int(op), int(rhs), int(literal))
                   for op, lhs, rhs, literal in rows[:-1]}
    op, _, node, _ = rows[-1]
    if op != QUERY or len(definitions) != len(rows) - 1:
        raise ValueError("Expected unique definitions followed by a single query")
    offsets = []
    visited = set()
    while node not in visited:
        visited.add(node)
        op, parent, literal = definitions[node]
        if op == BASE:
            answer = literal + sum(offsets)
            if not 0 <= answer <= 99:
                raise ValueError("The generated answer is outside the integer domain")
            return answer
        if op not in (PLUS, MINUS):
            raise ValueError("Unknown assignment operator")
        offsets.append(literal if op == PLUS else -literal)
        node = parent
    raise ValueError("The definitions contain a cycle")


def sample(size, index, rng):
    """Draw one full chain; each row contains type, lhs, rhs and unsigned literal."""
    if not 1 <= size < 64:
        raise ValueError("A chain needs between 1 and 63 operations")
    names = rng.permutation(64)[:size + 1]
    value = int(rng.integers(100))
    statements = [(BASE, names[0], MISSING, value)]
    for previous, current in zip(names[:-1], names[1:]):
        magnitude = int(rng.integers(1, 10))
        delta = magnitude if rng.integers(2) else -magnitude
        if not 0 <= value + delta <= 99:
            delta = -delta
        value += delta
        statements.append((PLUS if delta > 0 else MINUS, current, previous, magnitude))
    shuffled = np.asarray(statements, dtype=np.int64)[rng.permutation(len(statements))]
    query = np.asarray([[QUERY, MISSING, names[-1], 0]], dtype=np.int64)
    rows = np.concatenate([shuffled, query])
    return rows, reference(rows), {"steps": size, "statements": size + 1, "index": index}


def build():
    # The mutable registers are scalar value and fetched parent value. A base
    # ignores its predecessor. Every other statement adds its signed literal,
    # so after t loops all variables less than t links from the base are exact.
    constant, lhs, lhs_squared, rhs, is_base, is_minus, literal, value, fetched = range(9)
    width = 9
    kind = np.zeros((4, width))
    kind[:, constant] = 1
    kind[BASE, is_base] = kind[MINUS, is_minus] = 1
    left = np.zeros((65, width))
    left[:, lhs] = np.arange(65)
    left[:, lhs_squared] = np.arange(65)**2
    right = np.zeros_like(left)
    right[:, rhs] = np.arange(65)
    integers = np.zeros((100, width))
    integers[:, literal] = np.arange(100)

    # The dot product is maximized at lhs == rhs since it differs from
    # -2048*(rhs-lhs)^2 by a query-only constant. Every referenced id is unique.
    q = np.zeros((width, 2))
    k = np.zeros_like(q)
    q[rhs, 0], q[constant, 1] = 4096, -2048
    k[lhs, 0] = k[lhs_squared, 1] = 1
    v, o = np.zeros((width, 1)), np.zeros((1, width))
    v[value, 0] = o[0, fetched] = 1

    # Early transient values can be signed. Absolute magnitude is bounded by
    # 99 + 63*9, so the base gate suppresses either sign even during propagation.
    m = 1024
    units = [
        ({value: 1}, 0, {value: -1}),
        ({value: -1}, 0, {value: 1}),
        ({fetched: 1, is_base: -m}, 0, {value: 1}),
        ({fetched: -1, is_base: -m}, 0, {value: -1}),
        ({fetched: 1}, 0, {fetched: -1}),
        ({fetched: -1}, 0, {fetched: 1}),
        ({literal: 1}, 0, {value: 1}),
        ({literal: 1, is_minus: 100}, -100, {value: -2}),
    ]
    w1, w2 = np.zeros((width, len(units))), np.zeros((len(units), width))
    b1 = np.zeros(len(units))
    for j, (inputs, bias, outputs) in enumerate(units):
        for axis, gain in inputs.items():
            w1[axis, j] = gain
        b1[j] = bias
        for axis, gain in outputs.items():
            w2[j, axis] = gain
    classes = np.arange(100, dtype=np.float64)
    readout = np.zeros((width, len(classes)))
    readout[value] = 2 * classes
    spec = {"fields": ["type", "lhs", "rhs", "literal"], "layers": [1], "round_rule": "tokens"}
    arrays = {"e0": kind, "e1": left, "e2": right, "e3": integers,
              "l0.h0.q": q, "l0.h0.k": k, "l0.h0.v": v, "l0.h0.o": o,
              "l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": np.zeros(width),
              "readout": readout, "bias": -classes**2}
    return spec, arrays
