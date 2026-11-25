"""Clear a four-item combinatorial auction with Clarke pivot payments.

There is one unit of each item A, B, C and D. Each bidder submits a single
nonempty requested bundle and one integer value. Feasible winners have
disjoint bundles. The allocation maximizes reported total value, breaking
ties by smaller winner mask (bidder zero is the least significant bit).

The attention memory has one token per subset of bidders, up to 256 tokens.
Keys encode its reported value and subset ID. Values carry its reported
value and winner bits. The model queries this memory once for the full
auction and once excluding each bidder; these queries execute in parallel.
Input-dependent masks disallow overlapping bundles and the excluded bidder.
The two attention coefficients enforce exact bid priority and tie order.
Four output coefficients form the integer payment/status class from these
attention results. Fixed subset features, arithmetic and feasibility masks
are outside the stored parameter count. Exhaustive memory grows as 2**n.

The independent reference uses a dynamic program over occupied item sets.
See https://csd.cs.cmu.edu/sites/default/files/phd-thesis/CMU-CS-18-114.pdf
for single-minded bidders, welfare maximization and externality pricing.
"""
import numpy as np

ITEMS = "ABCD"
MAX_BID = 20
MAX_BIDDERS = 8
CLASSES = MAX_BID + 2


def encode(record):
    offers = record.get("bids") if isinstance(record, dict) else None
    if not isinstance(offers, list) or not 1 <= len(offers) <= MAX_BIDDERS:
        raise ValueError("Supply 1 through 8 bids in bidder order")
    tokens = []
    for offer in offers:
        if not isinstance(offer, dict):
            raise ValueError("Each bid needs a bundle and an integer value")
        bundle, value = offer.get("bundle"), offer.get("value")
        if not isinstance(bundle, str) or not bundle or bundle != "".join(item for item in ITEMS if item in bundle):
            raise ValueError("A bundle must list distinct items from ABCD in alphabetical order")
        if type(value) is not int or not 1 <= value <= MAX_BID:
            raise ValueError("Bid values must be integers from 1 through 20")
        tokens.append([int(item in bundle) for item in ITEMS] + [value])
    return np.asarray(tokens, dtype=np.int64)


def build_model():
    import torch
    from torch import nn

    class AuctionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # Even the largest possible mask penalty is smaller than a unit
            # welfare advantage. Distinct integer priorities differ by >=128.
            self.query = nn.Parameter(torch.tensor([32768., -128.]), requires_grad=False)
            self.readout = nn.Parameter(torch.tensor([1., -1., 1., 1.]), requires_grad=False)

        def forward(self, tokens):
            if tokens.ndim != 3 or tokens.shape[-1] != 5 or not 1 <= tokens.shape[1] <= MAX_BIDDERS:
                raise ValueError("Expected [batch,bidders,5] raw bid features")
            batch, n, _ = tokens.shape
            dtype = self.query.dtype
            bidders = torch.arange(n, device=tokens.device)
            masks = torch.arange(1 << n, device=tokens.device)
            members = ((masks[:, None] >> bidders) & 1).bool()
            selection = members.to(dtype)
            totals = selection @ tokens.to(dtype)
            feasible = (totals[:, :, :4] <= 1).all(dim=-1)
            welfare = totals[:, :, 4]
            keys = torch.stack((welfare, masks.to(dtype).expand_as(welfare)), dim=-1)
            logits = keys @ self.query
            # The unrestricted row asks who wins; the next n query without
            # each bidder. The empty allocation is available in every row.
            allowed = torch.cat((torch.ones_like(members.T[:1]), ~members.T), dim=0)
            probabilities = logits[:, None, :].masked_fill(~(feasible[:, None, :] & allowed), -torch.inf).softmax(-1)
            values = torch.cat((welfare[:, :, None], selection.expand(batch, -1, -1)), dim=-1)
            results = probabilities @ values
            winner = results[:, 0, 1:]
            # Winner price = optimal welfare without i - (full welfare - bid_i).
            # For a loser the two welfare terms cancel. The extra winner bit
            # reserves class zero for losing and class one for winning free.
            features = torch.stack((results[:, 1:, 0], results[:, :1, 0].expand(-1, n),
                                    winner * tokens[:, :, 4], winner), dim=-1)
            class_id = features @ self.readout
            labels = torch.arange(CLASSES, device=tokens.device, dtype=dtype)
            return -(class_id[:, :, None] - labels).abs()

    return AuctionTransformer().eval()


def reference(record):
    tokens = encode(record)
    bids = [(sum(int(used) << j for j, used in enumerate(row[:4])), int(row[4])) for row in tokens]

    def optimum(skip=None):
        # At a given occupied set keep the best (value, negative winner ID).
        states = {0: (0, 0)}
        for i, (bundle, bid) in enumerate(bids):
            if i == skip:
                continue
            updated = states.copy()
            for used, (value, neg_winners) in states.items():
                if bundle & used == 0:
                    combined = used | bundle
                    candidate = (value + bid, neg_winners - (1 << i))
                    if combined not in updated or candidate > updated[combined]:
                        updated[combined] = candidate
            states = updated
        best, neg_winners = max(states.values())
        return best, -neg_winners

    full, winners = optimum()
    answer = []
    for i, (_, bid) in enumerate(bids):
        if winners & (1 << i):
            without, _ = optimum(i)
            answer.append(1 + without - (full - bid))
        else:
            answer.append(0)
    return np.asarray(answer, dtype=np.int64)


def example():
    return {"bids": [{"bundle": "AB", "value": 17}, {"bundle": "A", "value": 10},
                     {"bundle": "B", "value": 9}, {"bundle": "C", "value": 5}]}


def generate(rng, size, index):
    if not 1 <= size <= MAX_BIDDERS:
        raise ValueError("Unsupported bidder count")
    symbols = list(rng.permutation(list(ITEMS)))

    def offer(items, amount):
        return {"bundle": "".join(sorted(items)), "value": int(amount)}

    def random_bid(items):
        mask = int(rng.integers(1, 1 << len(items)))
        return offer([item for j, item in enumerate(items) if mask & (1 << j)], rng.integers(1, 21))

    mode = index % 6
    if mode in (0, 1):
        bids = [random_bid(symbols) for _ in range(size)]
    elif mode == 2:
        # An exclusive item forces at least one zero-payment winner.
        bids = [offer(symbols[:1], rng.integers(1, 21))] + [random_bid(symbols[1:]) for _ in range(size-1)]
    elif mode == 3 or size < 3:
        # Identical bids make the tie rule observable. Extra bids use other items.
        pair = offer(symbols[:1], rng.integers(1, 21))
        bids = [pair.copy() for _ in range(min(size, 2))] + [random_bid(symbols[1:]) for _ in range(size-2)]
    else:
        # One two-item bundle competes with two individual-item bids. Extra
        # bids use the other two items and cannot alter this competition.
        a, b = (int(v) for v in rng.integers(4, 11, size=2))
        joint = a+b if mode == 4 else int(rng.integers(max(a, b)+1, a+b))
        bids = [offer(symbols[:2], joint), offer(symbols[:1], a), offer(symbols[1:2], b)]
        bids.extend(random_bid(symbols[2:]) for _ in range(size-3))
    return {"bids": [bids[int(i)] for i in rng.permutation(size)]}


def format_output(record, prediction):
    features = encode(record)
    if len(prediction) != len(features) or any(type(k) is not int or not 0 <= k < CLASSES for k in prediction):
        raise ValueError("Expected one valid outcome code per bidder")
    outcomes = [{"winner": k > 0, "payment": max(0, k - 1)} for k in prediction]
    return {"outcomes": outcomes, "welfare": sum(int(v) for v, k in zip(features[:, 4], prediction) if k > 0)}
