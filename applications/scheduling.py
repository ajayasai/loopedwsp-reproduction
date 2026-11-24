"""Single-resource bookings: flag each half-open interval that conflicts with another."""

import numpy as np


NAME = "Booking conflicts"
UNIT = "bookings"
MAX_BOOKINGS = 128
DAY = 1440
SIZES = (8, 16, 32, 64, 128)
DEV_SIZES = (8, 16)
COUNT = 64


def oracle(tokens):
    """Resource-independent pairwise check using the intersection length."""
    spans = np.asarray(tokens, dtype=np.int64)[:, :2]
    starts, ends = spans.T
    intersects = np.minimum(ends[:, None], ends) > np.maximum(starts[:, None], starts)
    np.fill_diagonal(intersects, False)
    return intersects.any(axis=1).astype(np.int64)


def _structured(size, mode, rng):
    """Build a day from varied appointment patterns and random gaps."""
    patterns = ("single", "touch", "duplicate", "nest", "overlap", "chain")
    first = ("touch", "duplicate", "nest", "overlap", "chain", "single")[mode]
    intervals = []
    cursor = 0
    while len(intervals) < size:
        remaining = size - len(intervals)
        if not intervals:
            kind = first
        elif mode == 0:
            kind = rng.choice(("single", "touch"))
        else:
            kind = rng.choice(patterns)
        a, b, c, d = map(int, rng.integers(1, 5, size=4))
        if remaining == 1 or kind == "single":
            local = [(0, a+b)]
        elif kind == "touch":
            local = [(0, a), (a, a+b+c)]
        elif kind == "duplicate":
            local = [(0, a+b), (0, a+b)]
        elif kind == "nest":
            local = [(0, a+b+c), (a, a+b)]
        elif kind == "chain" and remaining >= 3:
            local = [(0, a+b), (a, a+b+c), (a+b, a+b+c+d)]
        else:
            local = [(0, a+b), (a, a+b+c)]
        intervals.extend((cursor+start, cursor+end) for start, end in local)
        cursor += max(end for _, end in local) + int(rng.integers(0, 3))
    span = max(end for _, end in intervals)
    # Shift and sometimes reflect the whole day; both preserve its calendar semantics.
    if rng.integers(2):
        intervals = [(span-end, span-start) for start, end in intervals]
    offset = int(rng.integers(DAY-span+1))
    return [(start+offset, end+offset) for start, end in intervals]


def sample(size, index, rng):
    if not 1 <= size <= MAX_BOOKINGS:
        raise ValueError("Expected between 1 and 128 bookings")
    mode = index % 8
    if mode < 6:
        spans = _structured(size, mode, rng)
    else:
        # Random starts in either a busy window or across the entire day.
        span = min(DAY-25, max(48, size*4)) if mode == 6 else DAY-25
        offset = int(rng.integers(DAY-25-span+1))
        starts = offset + rng.integers(0, span+1, size=size)
        ends = starts + rng.integers(1, 26, size=size)
        spans = list(zip(starts, ends))
    raw = np.asarray(spans, dtype=np.int64)[rng.permutation(size)]
    tokens = np.column_stack((raw, np.arange(size, dtype=np.int64)))
    return tokens, oracle(tokens)


def build():
    # Immutable booking [start,end), ordinary shuffled row index, and the two
    # candidate intervals transported one row per shared-block invocation.
    spec = {"fields": ["start_minute", "end_minute", "row"],
            "layers": [2, 0], "round_rule": "tokens"}
    one, start, end, row, square = range(5)
    left, right = (5, 6, 7), (8, 9, 10)
    incoming_left, incoming_right = (11, 12, 13), (14, 15, 16)
    left_count, right_count, hit = 17, 18, 19
    width = 20
    a = {}

    def mat(key, *shape):
        a[key] = np.zeros(shape, dtype=np.float64)
        return a[key]

    starts, ends = mat("e0", DAY+1, width), mat("e1", DAY+1, width)
    minutes = np.arange(DAY+1)
    starts[:, [start, left[0], right[0]]] = minutes[:, None]
    ends[:, [end, left[1], right[1]]] = minutes[:, None]
    idx = mat("e2", MAX_BOOKINGS, width)
    ids = np.arange(MAX_BOOKINGS)
    idx[:, [row, left[2], right[2]]] = ids[:, None]
    idx[:, one], idx[:, square] = 1, ids**2

    # Distance-to-index attention selects the adjacent row (or the endpoint
    # when there is no adjacent row). This depends only on row coordinates.
    for j, shift, fields, dest in ((0, -1, left, incoming_left), (1, 1, right, incoming_right)):
        key = f"l0.h{j}."
        q, k = mat(key+"q", width, 2), mat(key+"k", width, 2)
        gain = 1024*np.sqrt(2)
        q[row, 0], q[one, :], k[row, 0], k[square, 1] = 2*gain, (2*shift*gain, -gain), 1, 1
        v, o = mat(key+"v", width, 3), mat(key+"o", 3, width)
        for c in range(3):
            v[fields[c], c], o[c, dest[c]] = 1, 1

    w1, b1, w2 = mat("l0.w1", width, 24), mat("l0.b1", 24), mat("l0.w2", 24, width)
    mat("l0.b2", width)
    for side, (fields, dest, total) in enumerate(((left, incoming_left, left_count),
                                                 (right, incoming_right, right_count))):
        # For integer x, ReLU(x)-ReLU(x-1) is exactly 1 iff x>0. Require
        # both strict endpoint comparisons and a different source row.
        identity = (row, dest[2]) if side == 0 else (dest[2], row)
        for cmp, (positive, negative) in enumerate(((end, dest[0]), (dest[1], start), identity)):
            unit = 6*side + 2*cmp
            w1[positive, unit:unit+2], w1[negative, unit:unit+2] = 1, -1
            b1[unit+1], w2[unit:unit+2, total] = -1, (1, -1)
        for c in range(3):
            new, old = 12 + 6*side + 2*c, 13 + 6*side + 2*c
            w1[dest[c], new], w1[fields[c], old] = 1, 1
            w2[new, [dest[c], fields[c]]] = (-1, 1)
            w2[old, fields[c]] = -1

    w1, b1, w2 = mat("l1.w1", width, 4), mat("l1.b1", 4), mat("l1.w2", 4, width)
    mat("l1.b2", width)
    for j, total in enumerate((left_count, right_count)):
        # On a previously clean booking, any candidate satisfying all three
        # tests latches hit. Simultaneous hits may give 2, also a conflict.
        w1[total, j], w1[hit, j], b1[j], w2[j, hit] = 1, -1, -2, 1
        w1[total, j+2], w2[j+2, total] = 1, -1
    readout, bias = mat("readout", width, 2), mat("bias", 2)
    readout[hit, 1], bias[0] = 1, 0.5
    return spec, a
