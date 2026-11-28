"""Independent solvers used to regrade saved predictions from the original JSON."""

from collections import defaultdict
from functools import lru_cache
from itertools import product


def isotopes(record):
    intensities = [0] * 16
    sources = [set() for _ in intensities]
    for ion in record["ions"]:
        chlorine, bromine = ion["cl"], ion["br"]
        atoms = chlorine + bromine
        # Sum over individual labeled atom assignments. Denominator four
        # gives weights 3,1 for chlorine and 2,2 for bromine.
        for heavy in product((0, 1), repeat=atoms):
            mass = ion["mz"] + 2 * sum(heavy)
            ways = 3 ** (chlorine - sum(heavy[:chlorine])) * 2**bromine
            intensities[mass - 400] += ion["abundance"] * ways * 4 ** (3 - atoms)
            sources[mass - 400].add(ion["mz"])
    if sum(intensities) != 64 * sum(ion["abundance"] for ion in record["ions"]):
        raise ValueError("The intensity integral changed")
    categories = []
    for value, masses in zip(intensities, sources, strict=True):
        groups = ["occupied_bins" if value else "empty_bins"]
        if len(masses) > 1:
            groups.append("overlap_from_distinct_masses")
        categories.append(groups)
    return intensities, categories


DAY_NAMES = (
    "Imix",
    "Ik'",
    "Akb'al",
    "K'an",
    "Chicchan",
    "Kimi",
    "Manik'",
    "Lamat",
    "Muluk",
    "Ok",
    "Chuwen",
    "Eb'",
    "B'en",
    "Ix",
    "Men",
    "Kib'",
    "Kab'an",
    "Etz'nab'",
    "Kawak",
    "Ajaw",
)
MONTHS = (
    "Pop",
    "Wo",
    "Sip",
    "Sotz'",
    "Sek",
    "Xul",
    "Yaxk'in",
    "Mol",
    "Ch'en",
    "Yax",
    "Sak",
    "Kej",
    "Mak",
    "K'ank'in",
    "Muwan",
    "Pax",
    "K'ayab'",
    "Kumk'u",
    "Wayeb'",
)


@lru_cache(maxsize=1)
def calendar_cycle():
    # Turn the three counters literally one day at a time, including the
    # exceptional five-day month. The model solves inverse cycle constraints.
    result = {}
    number, name, day, month = 4, 19, 8, 17
    for offset in range(18980):
        key = (number, DAY_NAMES[name], day, MONTHS[month])
        if key in result:
            raise ValueError("Calendar Round repeated early")
        result[key] = offset
        number = number + 1 if number < 13 else 1
        name = (name + 1) % 20
        day += 1
        if day == (5 if month == 18 else 20):
            day = 0
            month = (month + 1) % 19
    if (number, name, day, month) != (4, 19, 8, 17):
        raise ValueError("Calendar Round failed to return to its origin")
    return result


def calendar_round(record):
    cycle = calendar_cycle()
    labels, categories = [], []
    for date in record["dates"]:
        offset = cycle.get((date["number"], date["name"], date["day"], date["month"]))
        groups = ["incompatible" if offset is None else "compatible"]
        if date["month"] == "Wayeb'":
            groups.append("short_month")
        labels.extend((365, 365) if offset is None else divmod(offset, 365))
        categories.append(groups)
    return labels, categories


def overlap(a, b):
    """Test closed, axis-aligned segment boxes (one dimension can have zero extent)."""
    return all(
        max(min(p, q), min(r, s)) <= min(max(p, q), max(r, s))
        for (p, q), (r, s) in zip(zip(*a), zip(*b), strict=True)
    )


@lru_cache(maxsize=16)
def arm_templates(lengths):
    # Enumerate absolute headings instead of building the model's turn table.
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    solutions = defaultdict(list)
    for headings in product(range(4), repeat=4):
        changes = [(b - a) % 4 for a, b in zip((0, *headings[:-1]), headings, strict=True)]
        if 2 in changes:
            continue
        point = (0, 0)
        segments = []
        code = 0
        for heading, distance, delta in zip(headings, lengths, changes, strict=True):
            dx, dy = directions[heading]
            end = (point[0] + dx * distance, point[1] + dy * distance)
            segments.append((point, end))
            point = end
            code = 3 * code + {1: 0, 0: 1, 3: 2}[delta]
        if any(overlap(segments[i], segments[j]) for i, j in ((0, 2), (0, 3), (1, 3))):
            continue
        solutions[point].append((code, segments))
    return solutions


def arm(record):
    templates = arm_templates(tuple(record["lengths"]))
    obstacles = [(xy, xy) for xy in record["obstacles"]]
    labels, categories = [], []
    for target in record["targets"]:
        free = templates.get(tuple(target), [])
        allowed = [code for code, segs in free if not any(overlap(s, o) for s in segs for o in obstacles)]
        count = len(allowed)
        groups = ["multiple" if count > 1 else "unique" if count == 1 else "unreachable"]
        if free and not allowed:
            groups.append("all_poses_blocked")
        if allowed and len(allowed) < len(free):
            groups.append("some_poses_blocked")
        labels.extend((count, min(allowed, default=81)))
        categories.append(groups)
    return labels, categories


ORACLES = {"isotopes": isotopes, "calendar_round": calendar_round, "arm": arm}
