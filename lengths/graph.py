"""Directed-graph distances with analytic static-attention message passing."""

from collections import deque

import numpy as np


SIZES = (7, 10, 12, 24, 36, 48)
DEV_SIZES = (7, 10, 12)
COUNT = 96
NODE, EDGE, QUERY = 0, 1, 2
UNREACHABLE = 64
TOPOLOGY_SOURCE = "https://mbzuai-oryx.github.io/LoopedWSP/demo/graph_task.js"
BRIEF = "https://github.com/mbzuai-oryx/LoopedWSP/blob/main/prompts/07_graph_maps_network_brief.txt"


def _distance(size, edges, source, target):
    links = [[] for _ in range(size)]
    for tail, head in edges:
        links[tail].append(head)
    d = {source: 0}
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        if vertex == target:
            return d[vertex]
        for neighbor in links[vertex]:
            if neighbor not in d:
                d[neighbor] = d[vertex] + 1
                queue.append(neighbor)
    return UNREACHABLE


def _map(size, hops, unreachable, rng):
    """Adapt the public demo's branch, decoy, extra-edge, and reversal procedure."""
    while True:
        links = {(i, i+1) for i in range(hops)}
        for v in range(hops+1, size):
            outbound = rng.random() < 0.7
            u = int(rng.integers(v))
            links.add((u, v) if outbound else (v, u))
        extra = int(np.floor(size * (0.45 if size <= 12 else 0.55) + 0.5))
        for _ in range(extra * 4):
            if len(links) >= size - 1 + extra:
                break
            u, v = map(int, rng.integers(size, size=2))
            if u == v or (u, v) in links or (v, u) in links:
                continue
            links.add((u, v))
            if _distance(size, links, 0, hops) != hops:
                links.remove((u, v))
        if unreachable:
            u = int(rng.integers(hops))
            links.remove((u, u+1))
            links.add((u+1, u))
            if _distance(size, links, 0, hops) != UNREACHABLE:
                continue
        return links


def sample(size, index, rng):
    """Return only shuffled problem tokens to the model; BFS labels stay with the grader."""
    if size not in SIZES:
        raise ValueError(f"Unsupported graph node count: {size}")
    lo, hi = (1, 4) if size in DEV_SIZES else (8, 16)
    unreachable = (index % 4 == 3)
    reachable_index = index - index // 4
    hops = int(rng.integers(lo, hi+1)) if unreachable else lo + reachable_index % (hi-lo+1)
    edges = _map(size, hops, unreachable, rng)
    ids = rng.choice(64, size=size, replace=False).tolist()
    tokens = [(NODE, v, v) for v in ids]
    tokens.extend((EDGE, ids[u], ids[v]) for u, v in sorted(edges))
    rng.shuffle(tokens)
    tokens.append((QUERY, ids[0], ids[hops]))
    # Grade from the shuffled, relabeled token stream rather than the requested
    # design distance, so layout construction and scoring have separate checks.
    source, target = tokens[-1][1:]
    actual_edges = [(u, v) for kind, u, v in tokens if kind == EDGE]
    distance = _distance(64, actual_edges, source, target)
    if distance != (UNREACHABLE if unreachable else hops):
        raise ValueError("Relabeled graph failed its independent shortest-distance check")
    return np.asarray(tokens, dtype=np.int64), distance, {
        "distance": "unreachable" if distance == UNREACHABLE else distance,
        "nodes": size, "edges": len(edges),
    }


def build():
    """Represent finite distance k by 2**(-8*k), and an unreachable node by zero."""
    spec = {"fields": ["type", "id1", "id2"], "layers": [1, 1, 0, 0], "round_rule": "tokens"}
    one, edge, query, first, first2, second, second2, potential, tail, extra, avg, score = range(12)
    cut = 12
    width = cut + 64
    a = {}

    def matrix(name, *shape):
        a[name] = np.zeros(shape, dtype=np.float64)
        return a[name]

    def mlp(layer, units):
        prefix = f"l{layer}."
        w1, b1 = matrix(prefix+"w1", width, units), matrix(prefix+"b1", units)
        w2 = matrix(prefix+"w2", units, width)
        matrix(prefix+"b2", width)
        return w1, b1, w2

    types = matrix("e0", 3, width)
    types[:, one] = 1
    types[EDGE, edge] = 1
    types[QUERY, [query, potential]] = 1
    v = np.arange(64, dtype=np.float64)
    for name, coordinate, square in (("e1", first, first2), ("e2", second, second2)):
        table = matrix(name, 64, width)
        table[:, coordinate], table[:, square] = v, v**2

    # Layer 0. For id1, choose the matching node; a matching query token wins
    # instead and supplies the source potential 1. Edges cannot win this lookup.
    # Layer 1. Average every proposal with matching id2 except the query token.
    # There is always a matching node, and at most 63 incoming edges.
    for layer, coordinate, square in ((0, first, first2), (1, second, second2)):
        pre = f"l{layer}.h0."
        q, k = matrix(pre+"q", width, 2), matrix(pre+"k", width, 2)
        scale = 4096 * np.sqrt(2)
        q[coordinate, 0], q[one, 1] = 2*scale, scale
        k[coordinate, 0], k[square, 1] = 1, -1
        values, output = matrix(pre+"v", width, 1), matrix(pre+"o", 1, width)
        if layer == 0:
            k[edge, 1], k[query, 1] = -8192, 0.5
            values[potential, 0], output[0, tail] = 1, 1
        else:
            k[query, 1] = -8192
            values[tail, 0], values[extra, 0] = 1/256, 255/256
            output[0, avg] = 1

    w1, b1, w2 = mlp(0, 3)
    # Edge proposals divide the tail by 256. The self proposal of the node
    # remains at full strength, preserving the best previously known path.
    w1[tail, 0], w1[edge, 0], w2[0, extra] = 1, -1, 1
    for unit, field in ((1, potential), (2, score)):
        w1[field, unit], w2[unit, field] = 1, -1

    w1, b1, w2 = mlp(1, 67)
    powers = np.exp2(-8*np.arange(64, dtype=np.float64))
    for k, power in enumerate(powers):
        # For max proposal >= power the average is >= power/64. If the
        # maximum is smaller, every term is at most power/256. These gaps
        # permit exact threshold bits even when the mean itself is rounded.
        w1[avg, k], b1[k], w2[k, cut+k] = -192/power, 2, 1
    for unit, field in enumerate((tail, extra, avg), start=64):
        w1[field, unit], w2[unit, field] = 1, -1

    w1, b1, w2 = mlp(2, 128)
    for k, power in enumerate(powers):
        # bit[k] is 1 exactly when the best distance is <= k.
        w1[cut+k, k], b1[k] = -1, 1
        w2[k, score] = 1
        w2[k, potential] = power - (powers[k+1] if k < 63 else 0)
        w1[cut+k, k+64], w2[k+64, cut+k] = 1, -1

    w1, b1, w2 = mlp(3, 1)
    # Keep the query's value at 1 so its id1 is always a source. Its score
    # still comes from the id2 lookup and records the distance to the target.
    w1[query, 0], w1[potential, 0], w2[0, potential] = 1, -1, 1
    classes = np.arange(65, dtype=np.float64)
    readout, bias = matrix("readout", width, 65), matrix("bias", 65)
    readout[score] = -2*classes
    bias[:] = 2*64*classes - classes**2
    return spec, a
