"""Independent regraders using integer travel times, a text reader and legal card play."""

from functools import lru_cache


@lru_cache(maxsize=None)
def travel(dx, dy):
    # First integer tick at which a unit-speed circular wave reaches the receiver.
    squared = dx * dx + dy * dy
    return next(t for t in range(10) if t * t >= squared)


def quake(record):
    readings = [(x, y, t) for x, y, t in record["stations"] if t is not None]
    if not readings:
        raise ValueError("An event requires at least one measured arrival")
    labels, groups = [], []
    for y in range(7):
        for x in range(7):
            times = [(travel(abs(x - sx), abs(y - sy)), arrival) for sx, sy, arrival in readings]
            possible = [origin for origin in range(16) if all(origin + delay == t for delay, t in times)]
            if len(possible) > 1:
                raise ValueError("Multiple origin ticks at a single location")
            labels.append(possible[0] + 1 if possible else 0)
            groups.append(["compatible_location"] if possible else [])
    return labels, groups


# Index corresponds to the printed lowercase letter, a through z.
LETTERS = (
    "1",
    "12",
    "14",
    "145",
    "15",
    "124",
    "1245",
    "125",
    "24",
    "245",
    "13",
    "123",
    "134",
    "1345",
    "135",
    "1234",
    "12345",
    "1235",
    "234",
    "2345",
    "136",
    "1236",
    "2456",
    "1346",
    "13456",
    "1356",
)


def braille(record):
    cells = record["cells"]
    numeric = capital = False
    labels, groups = [], []
    for cell in cells:
        if cell in ("3456", "6", "56"):
            numeric = cell == "3456"
            capital = cell == "6"
            label, group = 0, ["indicator"]
        elif cell == "":
            numeric = capital = False
            label, group = 1, ["space"]
        else:
            letter = LETTERS.index(cell)
            if numeric and letter < 10:
                label, group = 54 + (letter + 1) % 10, ["digit"]
            else:
                label = 28 + letter if capital else 2 + letter
                group = ["uppercase" if capital else "lowercase"]
                numeric = capital = False
        labels.append(label)
        groups.append(group)
    return labels, groups


SEATS = "NESW"
RANKS = "23456789TJQKA"


def opening_values(deal):
    # Simulate one card at a time. Each player either maximizes or minimizes
    # the NS score, including the player who wins and leads the second trick.
    hands = tuple(tuple(deal["hands"][s]) for s in SEATS)
    trump = deal["trump"]
    leader = SEATS.index(deal["leader"])

    def search(remaining, first, trick, forced=None):
        if len(trick) == 4:
            led = trick[0][0]

            def priority(i):
                suit, rank = trick[i]
                return (suit == trump, suit == led, RANKS.index(rank))

            winner = (first + max(range(4), key=priority)) % 4
            return int(winner % 2 == 0) + (search(remaining, winner, ()) if any(remaining) else 0)
        player = (first + len(trick)) % 4
        hand = remaining[player]
        following = tuple(card for card in hand if trick and card[0] == trick[0][0])
        options = (forced,) if forced is not None else following or hand
        scores = []
        for card in options:
            updated = list(remaining)
            updated[player] = tuple(c for c in hand if c != card)
            scores.append(search(tuple(updated), first, (*trick, card)))
        return (max if player % 2 == 0 else min)(scores)

    return [search(hands, leader, (), card) for card in hands[leader]]


def bridge(record):
    labels, groups = [], []
    for deal in record["deals"]:
        values = opening_values(deal)
        score = (max if SEATS.index(deal["leader"]) % 2 == 0 else min)(values)
        best = [i for i, v in enumerate(values) if v == score]
        label = 4 * score + sum(1 << i for i in best)
        ranks = [RANKS.index(c[1]) for c in deal["hands"][deal["leader"]]]
        group = [
            "tied_openings" if len(best) == 2 else "unique_opening",
            "no_trump" if deal["trump"] == "NT" else "trump",
        ]
        if all(ranks[i] < max(ranks) for i in best):
            group.append("higher_opening_suboptimal")
        groups.append(group)
        labels.append(label)
    return labels, groups


ORACLES = {"quake": quake, "braille": braille, "bridge": bridge}
