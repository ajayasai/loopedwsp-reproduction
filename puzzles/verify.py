"""Independent streaming graders for the puzzle trajectories.

Rules and numbering: https://arxiv.org/html/2506.06941v1#A1.SS1
Comparison with LoopedWSP: https://mbzuai-oryx.github.io/LoopedWSP/ (Appendix E).

For Hanoi, transferring the n smallest disks between distinct pegs requires
T(n) >= 2*T(n-1)+1. To move disk n, every smaller disk must first clear its
source and destination. After its final move the smaller stack must reach it
again. The direct recursive transfer using the third peg attains equality,
so T(n) = 2**n - 1.

For checker jumping, all n*n red/blue pairs exchange order once. Each jump
exchanges one such pair. Both colors move forward by total distance n*(n+1)
each, so n*n jumps of two squares leave exactly 2*n single-square slides.
Every complete legal solution therefore uses n*(n+2) moves. One construction
alternates moving red and blue runs, starting with red. The run lengths are
1..n-1, then n three times, then n-1..1. In each run that color has a unique
legal source for the empty destination at every move.
"""

from collections import deque
import json
from operator import index


def _size(n):
    if isinstance(n, bool):
        raise ValueError("The puzzle size must be a nonnegative integer")
    n = index(n)
    if n < 0:
        raise ValueError("The puzzle size must be nonnegative")
    return n


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean move coordinates are invalid")
    return index(value)


def _result(n, count, legal, goal, optimum, error):
    valid = error is None and goal
    return {"size": n, "moves": count, "legal_moves": legal, "all_moves_legal": error is None,
            "valid": valid, "goal": goal, "optimal_moves": optimum, "optimal": valid and count == optimum,
            "first_error": error}


def verify_hanoi(n, moves):
    """Replay (disk, source, destination) triples. first_error.step counts from 1."""
    n = _size(n)
    goal_stack = list(range(n, 0, -1))
    pegs = [goal_stack.copy(), [], []]
    count = legal = 0
    error = None
    for count, move in enumerate(moves, 1):
        if error is not None:
            continue  # Still count every submitted move; only the legal prefix has a board.
        try:
            disk, src, dst = map(_integer, move)
        except (TypeError, ValueError, OverflowError):
            error = {"step": count, "reason": "expected three integer fields"}
            continue
        if not 0 <= src < 3 or not 0 <= dst < 3:
            reason = "peg out of range"
        elif src == dst:
            reason = "source and destination pegs coincide"
        elif not 1 <= disk <= n:
            reason = "disk out of range"
        elif not pegs[src]:
            reason = "source peg is empty"
        elif pegs[src][-1] != disk:
            reason = "specified disk is not topmost on the source peg"
        elif pegs[dst] and pegs[dst][-1] < disk:
            reason = "larger disk placed on a smaller disk"
        else:
            pegs[dst].append(pegs[src].pop())
            legal += 1
            continue
        error = {"step": count, "move": [disk, src, dst], "reason": reason}
    return _result(n, count, legal, pegs[2] == goal_stack, (1 << n) - 1, error)


def verify_checkers(n, moves):
    """Replay zero-based (source, destination) pairs, inferring color from the board."""
    n = _size(n)
    board = list("R"*n + "_" + "B"*n)
    count = legal = slides = jumps = 0
    error = None
    for count, move in enumerate(moves, 1):
        if error is not None:
            continue
        try:
            src, dst = map(_integer, move)
        except (TypeError, ValueError, OverflowError):
            error = {"step": count, "reason": "expected two integer fields"}
            continue
        distance = dst - src
        if not 0 <= src < len(board) or not 0 <= dst < len(board):
            reason = "board position out of range"
        elif board[src] == "_":
            reason = "source square is empty"
        elif board[dst] != "_":
            reason = "destination square is occupied"
        elif abs(distance) not in (1, 2):
            reason = "a move must slide one square or jump two"
        elif (board[src] == "R") != (distance > 0):
            reason = "checker moved backward"
        elif abs(distance) == 2 and board[(src + dst)//2] != ("B" if board[src]=="R" else "R"):
            reason = "jump must cross one opposite-colored checker"
        else:
            board[dst], board[src] = board[src], "_"
            legal += 1
            slides += abs(distance) == 1
            jumps += abs(distance) == 2
            continue
        error = {"step": count, "move": [src, dst], "reason": reason}
    goal = board == list("B"*n + "_" + "R"*n)
    return {**_result(n, count, legal, goal, n*(n+2), error), "slides": slides, "jumps": jumps}


def _checker_bfs(n):
    """A tiny-state shortest-path oracle, using four local string rewrites."""
    start, end = "R"*n + "_" + "B"*n, "B"*n + "_" + "R"*n
    parents = {start: None}
    queue = deque([start])
    while queue:
        board = queue.popleft()
        if board == end:
            moves = []
            while parents[board] is not None:
                board, move = parents[board]
                moves.append(move)
            return moves[::-1]
        for before, after, origin, dest in (("R_", "_R", 0, 1), ("_B", "B_", 1, 0),
                                             ("RB_", "_BR", 0, 2), ("_RB", "BR_", 2, 0)):
            i = board.find(before)
            if i >= 0:
                child = board[:i] + after + board[i+len(before):]
                if child not in parents:
                    parents[child] = (board, (i+origin, i+dest))
                    queue.append(child)
    raise AssertionError("Checker BFS did not reach the goal")


def self_test():
    """Exercise the graders only, independently of both transformer runners."""
    distances = {}
    for n in range(1, 6):
        path = _checker_bfs(n)
        report = verify_checkers(n, iter(path))
        assert report["valid"] and report["optimal"] and len(path) == n*(n+2)
        assert report["slides"] == 2*n and report["jumps"] == n*n
        distances[n] = len(path)
    negative = [
        (verify_hanoi, 2, [(1,-1,0)], "range"),
        (verify_hanoi, 2, [(1,0,0)], "coincide"),
        (verify_hanoi, 2, [(3,0,1)], "disk out"),
        (verify_hanoi, 2, [(1,1,2)], "empty"),
        (verify_hanoi, 2, [(2,0,1)], "topmost"),
        (verify_hanoi, 2, [(1,0,1), (2,0,1)], "smaller"),
        (verify_hanoi, 1, [(True,0,2)], "integer"),
        (verify_checkers, 2, [(-1,2)], "range"),
        (verify_checkers, 2, [(2,3)], "source square is empty"),
        (verify_checkers, 2, [(1,3)], "occupied"),
        (verify_checkers, 3, [(0,3)], "one square"),
        (verify_checkers, 2, [(1,2), (2,1)], "backward"),
        (verify_checkers, 2, [(0,2)], "opposite"),
        (verify_checkers, 2, [(1,2,3)], "integer"),
    ]
    for verifier,n,moves,reason in negative:
        res = verifier(n, iter(moves))
        assert not res["valid"] and res["first_error"]["step"] == len(moves)
        assert reason in res["first_error"]["reason"] and res["legal_moves"] == len(moves)-1
    incomplete = verify_hanoi(2, [(1,0,1)])
    assert incomplete["all_moves_legal"] and not incomplete["goal"] and not incomplete["valid"]
    solved = verify_hanoi(2, [(1,0,1), (2,0,2), (1,1,2)])
    assert solved["valid"] and solved["optimal"] and solved["first_error"] is None
    detour = verify_hanoi(1, [(1,0,1), (1,1,2)])
    assert detour["valid"] and not detour["optimal"]
    bad_tail = verify_hanoi(1, [(1,0,2), (1,0,1), (1,2,1)])
    assert bad_tail["moves"] == 3 and bad_tail["legal_moves"] == 1 and not bad_tail["valid"]
    return {"checker_bfs_distances": distances, "illegal_rule_cases": len(negative),
            "hanoi_completion_checks": 4}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
