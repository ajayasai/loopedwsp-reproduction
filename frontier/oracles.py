"""Regrade three domains directly from the public JSON input, without neural weights."""


def contours(record):
    level = record["level"]
    heights = record["heights"]
    ordinary = {(0, 1): 1, (1, 2): 2, (2, 3): 3, (0, 3): 4, (0, 2): 5, (1, 3): 6}
    labels, groups = [], []
    for top, bottom in zip(heights, heights[1:]):
        for col in range(len(top) - 1):
            a, b, c, d = (v - level for v in (top[col], top[col + 1], bottom[col + 1], bottom[col]))
            if 0 in (a, b, c, d):
                raise ValueError("A vertex lies exactly on the isoline")
            marked = tuple(i for i, (x, y) in enumerate(((a, b), (b, c), (c, d), (d, a))) if x * y < 0)
            group = []
            if len(marked) == 4:
                det = a * c - b * d
                label = 7 if det > 0 else 8 if det < 0 else 9
                group.append("four_edge_cells")
                if det == 0:
                    group.append("exact_saddles")
                # The arithmetic mean gives a common shortcut. Compare only
                # where both this shortcut and the bilinear rule choose a pairing.
                middle = a + b + c + d
                if det and middle:
                    group.append("center_comparable")
                    if (7 if a * middle > 0 else 8) != label:
                        group.append("center_wrong")
            else:
                label = ordinary[marked] if marked else 0
            labels.append(label)
            groups.append(group)
    return labels, groups


TRAITS = ("plain", "barred", "spotted")


def plumage(record):
    tips, joins = record["tips"], record["joins"]
    n = len(tips)
    graph = [[] for _ in range(n + len(joins))]
    for parent, (a, b) in enumerate(joins, n):
        for child in (a, b):
            graph[child].append(parent)
            graph[parent].append(child)

    # Reroot the undirected tree on each focal ancestor. This puts every
    # observation below that node and avoids an outside-message recurrence.
    def scores(node, parent):
        if node < n:
            return [0 if tips[node] == trait else len(graph) + 1 for trait in TRAITS]
        children = [scores(neighbor, node) for neighbor in graph[node] if neighbor != parent]
        return [
            sum(min(cost + int(color != state) for color, cost in enumerate(child)) for child in children)
            for state in range(3)
        ]

    def mask(costs):
        return sum(1 << i for i, value in enumerate(costs) if value == min(costs))

    labels, groups = [], []
    for node, (a, b) in enumerate(joins, n):
        label = mask(scores(node, None))
        parent = next((p for p in graph[node] if p not in (a, b)), None)
        descendant_only = mask(scores(node, parent))
        group = []
        if label.bit_count() > 1:
            group.append("ambiguous_ancestors")
        if label == 7:
            group.append("all_three_states")
        if label != descendant_only:
            group.append("outside_changes_set")
        labels.append(label)
        groups.append(group)
    return labels, groups


def clifford(record):
    # Enumerate the eight initial stabilizers, then conjugate them forward.
    # i^phase X^x Z^z uses two 3-bit strings and an exponent modulo four.
    stabilizers = {}
    for initial_z in range(8):
        x, z, phase = 0, initial_z, 0
        for gate in record["gates"]:
            symbol, site = gate[:2]
            bit = 1 << site
            xb, zb = int(bool(x & bit)), int(bool(z & bit))
            if symbol == "H":
                phase += 2 * xb * zb
                if xb != zb:
                    x ^= bit
                    z ^= bit
            elif symbol in ("S", "Sdg"):
                phase += xb if symbol == "S" else -xb
                if xb:
                    z ^= bit
            elif symbol == "CX":
                target = 1 << gate[2]
                if xb:
                    x ^= target
                if z & target:
                    z ^= bit
            else:
                raise ValueError(f"Unsupported gate {symbol}")
        stabilizers[x, z] = phase % 4

    def expectation(word):
        x = sum(1 << i for i, s in enumerate(word) if s in "XY")
        z = sum(1 << i for i, s in enumerate(word) if s in "ZY")
        if (x, z) not in stabilizers:
            return 0
        residual = (stabilizers[x, z] - word.count("Y")) % 4
        if residual not in (0, 2):
            raise ValueError("A stabilizer should have a real sign")
        return 1 if residual == 0 else -1

    labels, groups = [], []
    for word in record["observables"]:
        value = expectation(word)
        group = [{-1: "negative", 0: "random", 1: "positive"}[value]]
        singles = ["I" * i + axis + "I" * (2 - i) for i, axis in enumerate(word) if axis != "I"]
        if value and len(singles) > 1 and any(expectation(single) == 0 for single in singles):
            group.append("definite_joint_random_marginal")
        labels.append(value + 1)
        groups.append(group)
    return labels, groups


ORACLES = {"contours": contours, "plumage": plumage, "clifford": clifford}
