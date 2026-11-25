"""Pick a film-club screening by applying a tied attention block to ranked ballots.

A record contains five distinct nonempty film titles in nomination order and
1..128 ballots. Every ballot ranks all five titles exactly once, best first.
Each round transfers every vote to its highest-ranked remaining film, then
eliminates the film with the fewest votes. An elimination tie goes to the
earliest nomination in the tie. Count to the last remaining film, including
when a film has already secured a majority. The complete output gives four
eliminated film IDs followed by the winner, all in the original nomination ID
space. The public formatter translates those IDs back into titles.

Each of four applications of the same block first attends to positions in
each ballot, with active films ahead of inactive films and better ranks first.
A uniform attention pool across voters yields support counts after rescaling.
A second attention reads the lowest-count active candidate, breaking ties by
nomination ID. Subtracting its ID distribution updates the active state.
Attention performs the soft selection in both stages. Only the evaluator
uses argmax when decoding class scores. The block has five stored scalar
attention-query coefficients. One-hot candidate values, positional features,
the uniform pool, and the residual subtraction are fixed operations.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FILMS = ("Moonlit Post Office", "Paper Comet", "Moth Orchestra",
         "The Clockwork Orchard", "Lantern Atlas")
CANDIDATES = len(FILMS)
MAX_VOTERS = 128


def encode(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a film-club record with films and ballots")
    films, ballots = record.get("films"), record.get("ballots")
    if (not isinstance(films, list) or len(films) != CANDIDATES or
            any(not isinstance(f, str) or not f.strip() for f in films) or len(set(films)) != CANDIDATES):
        raise ValueError("Expected five distinct, nonempty film titles in nomination order")
    if not isinstance(ballots, list) or not 1 <= len(ballots) <= MAX_VOTERS:
        raise ValueError(f"Expected between 1 and {MAX_VOTERS} ballots")
    ids = {title: i for i, title in enumerate(films)}
    out = np.empty((len(ballots), CANDIDATES), dtype=np.int64)
    for i, ranking in enumerate(ballots):
        if (not isinstance(ranking, list) or len(ranking) != CANDIDATES or
                any(not isinstance(title, str) for title in ranking) or set(ranking) != ids.keys()):
            raise ValueError("Each ballot must rank all five nominated films exactly once")
        out[i] = [ids[title] for title in ranking]
    return out


class BallotTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # The active-film bonus exceeds the full within-ballot rank range.
        # The vote coefficient exceeds the full nomination-order range.
        gap = 32.0
        self.preference_query = nn.Parameter(torch.tensor([gap, CANDIDATES*gap]), requires_grad=False)
        self.elimination_query = nn.Parameter(torch.tensor([-CANDIDATES*gap, -gap, CANDIDATES*gap]),
                                              requires_grad=False)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != CANDIDATES or not 1 <= tokens.shape[1] <= MAX_VOTERS:
            raise ValueError("Expected encoded ballots [batch, voters, 5]")
        batch, voters, films = tokens.shape
        ids = F.one_hot(tokens, num_classes=films).to(self.preference_query.dtype)
        order = torch.arange(films, device=tokens.device, dtype=ids.dtype)
        quality = (films - 1 - order).expand(batch, voters, -1)
        nomination = order.expand(batch, -1)
        active = ids.new_ones((batch, films))
        # Each voter gets equal mass in this pooling attention head.
        pool = ids.new_zeros((batch, 1, voters)).softmax(dim=-1)
        identity_values = torch.eye(films, device=tokens.device, dtype=ids.dtype)
        output = []
        for _ in range(films - 1):
            live_in_ballot = (ids * active[:, None, None, :]).sum(dim=-1)
            choice_keys = torch.stack((quality, live_in_ballot), dim=-1)
            support = ((choice_keys @ self.preference_query).softmax(dim=-1).unsqueeze(-2) @ ids).squeeze(-2)
            votes = (pool @ support).squeeze(1) * voters
            # Even a unanimous active candidate ranks above a removed one for
            # eligibility. The smallest support and then smallest ID exit.
            elimination_keys = torch.stack((votes, nomination, active*(voters+1)), dim=-1)
            scores = elimination_keys @ self.elimination_query
            output.append(scores)
            eliminated = scores.softmax(dim=-1) @ identity_values
            active = active - eliminated
        output.append(active)
        return torch.stack(output, dim=1)


def build_model():
    return BallotTransformer().eval()


def reference(record):
    """Direct integer tally over title strings, independent of tensor inference."""
    titles = record["films"]
    remaining = list(titles)
    result = []
    while len(remaining) > 1:
        counts = dict.fromkeys(remaining, 0)
        for ranking in record["ballots"]:
            favorite = next(title for title in ranking if title in counts)
            counts[favorite] += 1
        # Remaining titles preserve nomination order; min resolves ties by first occurrence.
        eliminated = min(remaining, key=counts.__getitem__)
        result.append(titles.index(eliminated))
        remaining.remove(eliminated)
    return np.asarray([*result, titles.index(remaining[0])], dtype=np.int64)


def format_output(record, prediction):
    films = record["films"]
    if len(prediction) != CANDIDATES or sorted(prediction) != list(range(CANDIDATES)):
        raise ValueError("Expected four eliminations and one winner, each nominated film exactly once")
    return {"eliminated": [films[i] for i in prediction[:-1]], "winner": films[prediction[-1]]}


def example():
    # A leads 3-2-2-1-0 on first choices. The third and fourth blocs transfer to B.
    rankings = [[0,1,2,3,4]]*3 + [[1,0,4,2,3]]*2 + [[2,1,3,4,0]]*2 + [[3,1,0,2,4]]
    return {"films": list(FILMS), "ballots": [[FILMS[j] for j in row] for row in rankings]}


def generate(rng, size, index):
    """Rotate uniform preferences, tied zero counts, transferred wins, and consensus."""
    if size not in (4, 8, 16, 32):
        raise ValueError("Generated elections use 4, 8, 16 or 32 voters")
    titles = rng.permutation(FILMS).tolist()
    candidates = rng.permutation(CANDIDATES).tolist()
    mode = index % 4
    if mode == 2 and size == 4:
        # The two-vote first-place film has the earliest ID of the three
        # supported films. Two transferred votes defeat it on the final tie.
        a, b, c = sorted(candidates[:3])
        prefixes = [[a], [a], [b, c], [c, b]]
    elif mode == 2:
        a, b, c, d, _ = candidates
        unit = size // 8
        prefixes = [[a]]*(3*unit) + [[b]]*(2*unit) + [[c, b]]*(2*unit) + [[d, b]]*unit
    elif mode == 1:
        # Two nominees have zero original choices, forcing an elimination tie.
        prefixes = [[candidates[i % 3]] for i in range(size)]
    elif mode == 3:
        prefixes = [[candidates[0]] for _ in range(size)]
    else:
        prefixes = [[] for _ in range(size)]
    rankings = []
    for prefix in prefixes:
        suffix = rng.permutation([i for i in range(CANDIDATES) if i not in prefix]).tolist()
        rankings.append([titles[j] for j in [*prefix, *suffix]])
    return {"films": titles, "ballots": [rankings[i] for i in rng.permutation(size)]}
