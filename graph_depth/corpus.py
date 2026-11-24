"""Deterministic matched pairs for pointer chasing and directed reachability."""

from collections import deque
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


N = 64
HOPS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)
DEV_HOPS = (1, 2, 4)
SEED = 20261124
SPLITS = (("dev", DEV_HOPS, 8), ("test", HOPS, 64))


def pointer_pair(k, pair_index, rng):
    # The first k edges form a simple path. Vertex k+1 donates the
    # replacement endpoint, which also lies outside the shared prefix.
    successor = np.empty(N, dtype=np.uint8)
    successor[:k] = np.arange(1, k + 1)
    donor = k + 1
    donor_target = int(rng.integers(k + 1, N))
    successor[donor] = donor_target
    free_rows = np.delete(np.arange(k, N), 1)
    if pair_index % 2 == 0:
        used = np.zeros(N, dtype=bool)
        used[1:k + 1] = True
        used[donor_target] = True
        successor[free_rows] = rng.permutation(np.flatnonzero(~used))
    else:
        successor[free_rows] = rng.integers(N, size=len(free_rows))
        if np.unique(successor).size == N:
            successor[free_rows[0]] = successor[free_rows[1]]

    # Both samples retain identical row order and the same randomly named source.
    names = rng.permutation(N).astype(np.uint8)
    order = rng.permutation(N)
    changed = successor.copy()
    changed[k - 1], changed[donor] = successor[donor], successor[k - 1]
    tables = np.stack([np.column_stack((names[order], names[mapping[order]]))
                       for mapping in (successor, changed)])
    labels = names[[k, donor_target]]
    if rng.integers(2):
        tables = tables[::-1]
        labels = labels[::-1]
    return tables, names[[0, 0]], labels


def reachability_pair(k, rng):
    # Two separate directed spines each have one forward-only cut. Additional
    # spine edges point backward inside a cut side, preserving the positive distance.
    adj = np.zeros((N, N), dtype=bool)
    main = np.arange(k + 1)
    donor_size = int(rng.integers(3, min(N - len(main) - 3, 18) + 1))
    donor = np.arange(len(main), len(main) + donor_size)
    cuts = []
    for spine in (main, donor):
        adj[spine[:-1], spine[1:]] = True
        cut = int(rng.integers(1, len(spine)))
        cuts.append((spine[cut - 1], spine[cut]))
        for segment in (spine[:cut], spine[cut:]):
            for pos in range(1, len(segment)):
                if rng.random() < 0.45:
                    adj[segment[pos], segment[int(rng.integers(pos))]] = True

    # Side trees can branch from other side-tree vertices. The first two
    # leaves make directed cycles and branch off different components.
    first_spare = int(donor[-1]) + 1
    for leaf in range(first_spare, N):
        if leaf == first_spare:
            parent = int(rng.choice(main[:-1]))
        elif leaf == first_spare + 1:
            parent = int(rng.choice(donor[:-1]))
        else:
            parent = int(rng.integers(leaf))
        adj[parent, leaf] = True
        if leaf <= first_spare + 1 or rng.random() < 0.7:
            adj[leaf, parent] = True

    (u, v), (a, b) = cuts
    negative = adj.copy()
    negative[u, v] = negative[a, b] = False
    negative[u, b] = negative[a, v] = True
    names = rng.permutation(N).astype(np.uint8)
    order = np.argsort(names)
    matrices = np.stack([graph[np.ix_(order, order)] for graph in (adj, negative)])
    labels = np.array([1, 0], dtype=np.uint8)
    if rng.integers(2):
        matrices = matrices[::-1]
        labels = labels[::-1]
    return matrices, names[[0, 0]], names[[k, k]], labels


def distance(adj, source, target):
    seen = {int(source)}
    queue = deque([(int(source), 0)])
    while queue:
        node, hops = queue.popleft()
        if node == target:
            return hops
        for dest in np.flatnonzero(adj[node]):
            if int(dest) not in seen:
                seen.add(int(dest))
                queue.append((int(dest), hops + 1))
    return -1


def check_pointer(k, pair_index, tables, sources, labels):
    if sources[0] != sources[1] or labels[0] == labels[1] or not np.array_equal(tables[0, :, 0], tables[1, :, 0]):
        raise ValueError("Pointer pair changed its query, row order, or retained its endpoint")
    if not np.array_equal(np.sort(tables[0, :, 0]), np.arange(N)):
        raise ValueError("Pointer table must contain each source ID once")
    mapped = tables[:, np.argsort(tables[0, :, 0]), 1]
    counts = [np.bincount(row, minlength=N) for row in mapped]
    if not np.array_equal(*counts) or np.count_nonzero(mapped[0] != mapped[1]) != 2:
        raise ValueError("Pointer pair does not have the matched two-edge swap")
    if bool(np.all(counts[0] == 1)) != (pair_index % 2 == 0):
        raise ValueError("Pointer cohort does not match its advertised function type")
    paths = []
    for successor, source, label in zip(mapped, sources, labels):
        path = [int(source)]
        for _ in range(k):
            path.append(int(successor[path[-1]]))
        if len(set(path)) != k + 1 or path[-1] != label:
            raise ValueError("Pointer label or simple path length is incorrect")
        paths.append(path)
    if paths[0][:-1] != paths[1][:-1]:
        raise ValueError("Pointer interventions must change exactly the final hop of the shared prefix")


def check_reachability(k, matrices, sources, targets, labels):
    if sources[0] != sources[1] or targets[0] != targets[1] or set(labels) != {0, 1}:
        raise ValueError("Reachability pair must share its query and have opposite labels")
    a, b = matrices
    if (np.count_nonzero(a != b) != 4 or not np.array_equal(a.sum(0), b.sum(0))
            or not np.array_equal(a.sum(1), b.sum(1))):
        raise ValueError("Reachability pair must preserve all vertex degrees")
    for adj, source, target, label in zip(matrices, sources, targets, labels):
        if distance(adj, source, target) != (k if label else -1):
            raise ValueError("BFS disagrees with the reachability label or distance cohort")
        if not np.any(adj & adj.T) or adj.sum(1).max() < 2:
            raise ValueError("Directed reachability graph lacks a cycle or branch")


def write_cases(path):
    """Write the cases archive and return its machine-independent per-array manifest."""
    arrays = {}
    for split_id, (split, hops, pairs) in enumerate(SPLITS):
        for k in hops:
            for task_id, task in enumerate(("pointer", "reachability")):
                seed = np.random.SeedSequence([SEED, split_id, task_id, k])
                rng = np.random.Generator(np.random.PCG64(seed))
                items = []
                for i in range(pairs):
                    if task == "pointer":
                        record = pointer_pair(k, i, rng)
                        check_pointer(k, i, *record)
                        fields = ("table", "source", "label")
                    else:
                        record = reachability_pair(k, rng)
                        check_reachability(k, *record)
                        fields = ("adj", "source", "target", "label")
                    items.append(record)
                for field, values in zip(fields, zip(*items)):
                    arrays[f"{task}_{split}_{k}_{field}"] = np.concatenate(values)
    manifest = {"vertices": N, "seed": SEED, "bit_generator": "PCG64",
                "seed_coordinates": ["seed", "split_index", "task_index", "hops"],
                "split_order": ["dev", "test"], "task_order": ["pointer", "reachability"],
                "pairs_per_bin": {split: count for split, _, count in SPLITS},
                "hops": {split: list(hops) for split, hops, _ in SPLITS},
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                 "sha256": sha256(value.tobytes(order="C")).hexdigest()}
                           for key, value in sorted(arrays.items())}}
    np.savez_compressed(path, **arrays)
    return manifest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    manifest = write_cases(here / "cases.npz")
    (here / "cases-manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("ascii"))
    print(f"Wrote {len(manifest['arrays'])} checked graph-depth arrays.")
