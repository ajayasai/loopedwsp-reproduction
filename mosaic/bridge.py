"""Two-trick double-dummy bridge with attention for winners and optimal replies."""

import numpy as np
import torch
from torch import nn


SEATS = "NESW"
SUITS = "CDHS"
RANKS = "23456789TJQKA"
DECK = tuple(suit + rank for suit in SUITS for rank in RANKS)


def encode(record):
    if not isinstance(record, dict) or set(record) != {"deals"}:
        raise ValueError("Supply a deals list")
    deals = record["deals"]
    if not isinstance(deals, list) or not 1 <= len(deals) <= 64:
        raise ValueError("Supply 1 to 64 two-trick positions")
    rows = []
    for deal in deals:
        if not isinstance(deal, dict) or set(deal) != {"hands", "leader", "trump"}:
            raise ValueError("Each deal needs hands, leader and trump")
        hands = deal["hands"]
        if not isinstance(hands, dict) or set(hands) != set(SEATS):
            raise ValueError("Supply hands for N, E, S and W")
        if any(not isinstance(hand, list) or len(hand) != 2 for hand in hands.values()):
            raise ValueError("Give each seat exactly two remaining cards")
        cards = [card for seat in SEATS for card in hands[seat]]
        if any(card not in DECK for card in cards) or len(set(cards)) != 8:
            raise ValueError("Use eight distinct cards, such as SA or DT")
        if deal["leader"] not in tuple(SEATS) or deal["trump"] not in (*SUITS, "NT"):
            raise ValueError("Use N/E/S/W for the leader and C/D/H/S/NT for the trump")
        rows.append(
            [SEATS.index(deal["leader"]), (*SUITS, "NT").index(deal["trump"])]
            + [DECK.index(c) for c in cards]
        )
    return np.asarray(rows, dtype=np.int64)


class EndplayTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.suit_bonuses = nn.Parameter(torch.tensor([16.0, 32.0]), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(128.0), requires_grad=False)

    def winner(self, cards, led, trump):
        suits = cards // 13
        score = (
            cards % 13 + (suits == led) * self.suit_bonuses[0] + (suits == trump) * self.suit_bonuses[1]
        )
        return (self.sharpness * score).softmax(-1)

    def forward(self, tokens):
        batch, deals, _ = tokens.shape
        raw = tokens.reshape(-1, 10)
        device = raw.device
        players = (raw[:, :1] + torch.arange(4, device=device)) % 4
        ns = players.remainder(2) == 0
        hands = raw[:, 2:].reshape(-1, 4, 2).gather(1, players[:, :, None].expand(-1, -1, 2))
        branches = torch.arange(16, device=device)[:, None]
        first_choice = ((branches >> torch.arange(3, -1, -1, device=device)) & 1).bool()
        first = torch.where(first_choice, hands[:, None, :, 1], hands[:, None, :, 0])
        second = torch.where(first_choice, hands[:, None, :, 0], hands[:, None, :, 1])

        trump = raw[:, 1, None, None]
        led = first[:, :, :1] // 13
        # If my other card follows suit, discarding an off-suit card is illegal.
        legal = ((first // 13) == led) | ((second // 13) != led)
        first_winner = self.winner(first, led, trump)
        second_led = (first_winner * (second // 13)).sum(-1, keepdim=True)
        second_winner = self.winner(second, second_led, trump)
        value = ((first_winner + second_winner) * ns[:, None, :]).sum(-1)

        # The 16 branches are ordered by the four successive binary choices.
        # Merge each pair of legal replies before evaluating the previous seat.
        for seat in (3, 2, 1):
            values = value.reshape(len(raw), -1, 2)
            allowed = legal[:, :: 2 ** (3 - seat), seat].reshape_as(values)
            sign = torch.where(ns[:, seat], 1.0, -1.0)[:, None, None]
            scores = (self.sharpness * sign * values).masked_fill(~allowed, -torch.inf)
            value = (scores.softmax(-1) * values).sum(-1)
        best = torch.where(ns[:, :1], value.amax(-1, keepdim=True), value.amin(-1, keepdim=True))
        tied = value == best
        label = 4 * best.squeeze(-1) + tied[:, 0] + 2 * tied[:, 1]
        classes = torch.arange(12, dtype=self.sharpness.dtype, device=device)
        return -(label.reshape(batch, deals, 1) - classes).abs()


def build_model():
    return EndplayTransformer().eval()


def _opening_values(deal):
    """Play the actual eight-card game depth-first, with the winner leading next."""
    original = [tuple(deal["hands"][seat]) for seat in SEATS]
    leader = SEATS.index(deal["leader"])
    trump = deal["trump"]

    def play(hands, seat, trick):
        led = trick[0][1][0] if trick else None
        choices = [c for c in hands[seat] if c[0] == led] or hands[seat]
        values = []
        for card in choices:
            next_hands = hands.copy()
            next_hands[seat] = tuple(c for c in hands[seat] if c != card)
            played = [*trick, (seat, card)]
            if len(played) < 4:
                value = play(next_hands, (seat + 1) % 4, played)
            else:
                winner = max(
                    played,
                    key=lambda p: (p[1][0] == trump, p[1][0] == played[0][1][0], RANKS.index(p[1][1])),
                )[0]
                value = int(winner % 2 == 0)
                if any(next_hands):
                    value += play(next_hands, winner, [])
            values.append(value)
        return (min if seat % 2 else max)(values)

    opening = []
    for card in original[leader]:
        hands = original.copy()
        hands[leader] = tuple(c for c in hands[leader] if c != card)
        opening.append(play(hands, (leader + 1) % 4, [(leader, card)]))
    return opening


def reference(record):
    encode(record)
    labels = []
    for deal in record["deals"]:
        options = _opening_values(deal)
        best = (max if deal["leader"] in "NS" else min)(options)
        labels.append(4 * best + sum(1 << i for i, v in enumerate(options) if v == best))
    return np.asarray(labels, dtype=np.int64)


def generate(rng, size, index):
    deals = []
    for i in range(size):
        # Restrict some boards to one or two suits to exercise forced follows
        # and interactions within long suits. The remaining boards use all four.
        n_suits = (1, 2, 4, 4)[(i + index) % 4]
        suits = rng.choice(list(SUITS), size=n_suits, replace=False)
        deck = [suit + rank for suit in suits for rank in RANKS]
        cards = rng.choice(deck, size=8, replace=False).tolist()
        deal = {
            "hands": {s: cards[2 * j : 2 * j + 2] for j, s in enumerate(SEATS)},
            "leader": str(rng.choice(list(SEATS))),
            "trump": "NT" if (i + index) % 3 == 0 else str(rng.choice(suits)),
        }
        deals.append(deal)
    return {"deals": deals}


def example():
    hands = {"N": ["CK", "S6"], "E": ["H3", "C8"], "S": ["C3", "D2"], "W": ["HJ", "CA"]}
    return {"deals": [{"hands": hands, "leader": "N", "trump": trump} for trump in ("NT", "H")]}


def format_output(record, prediction):
    result = []
    if len(prediction) != len(record["deals"]):
        raise ValueError("Expected one class per deal")
    for value, deal in zip(prediction, record["deals"], strict=True):
        tricks, mask = divmod(int(value), 4)
        if not 0 <= tricks <= 2 or mask == 0:
            raise ValueError("Invalid trick count or opening-lead mask")
        hand = deal["hands"][deal["leader"]]
        result.append(
            {
                "ns_tricks": tricks,
                "optimal_leads": [card for i, card in enumerate(hand) if mask & (1 << i)],
            }
        )
    return {"solutions": result}
