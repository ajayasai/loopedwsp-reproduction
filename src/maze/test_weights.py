"""Handmade component and end-to-end checks, independent of benchmark data."""

import unittest

import numpy as np

from looped import dense_numpy
from weights import D_HEAD, DIRECTIONS, build_weights


def ids_for(weights, rows):
    symbols = weights["chars"]
    return np.array([[symbols.index(char) for char in "".join(rows)]])


def output_for(weights, ids, rounds):
    indices = dense_numpy(weights, ids, rounds).argmax(axis=-1)[0]
    return "".join(weights["outputs"][i] for i in indices)


class MazeWeightsTest(unittest.TestCase):
    def test_initial_readout(self):
        rows = ["S ", "#G"]
        weights = build_weights(2)
        self.assertEqual(output_for(weights, ids_for(weights, rows), 0), "".join(rows))

    def test_static_attention_selectors(self):
        # Check all four edges and corners. A query beyond the grid must read
        # its own edge cell. The test explicitly checks floating-point softmax.
        for side in (3, 30):
            weights = build_weights(side)
            embeddings = weights["token"][0] + weights["row"][:, None] + weights["col"][None, :]
            embeddings = embeddings.reshape(side * side, -1)
            layer = weights["layers"][0]
            y, x = np.indices((side, side))
            for dtype in (np.float32, np.float64):
                h = embeddings.astype(dtype)
                for direction, (dy, dx) in enumerate(DIRECTIONS):
                    q = h @ layer["q"][direction].astype(dtype)
                    k = h @ layer["k"][direction].astype(dtype)
                    scores = (q @ k.T) / dtype(np.sqrt(D_HEAD))
                    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    target = np.clip(y + dy, 0, side - 1) * side + np.clip(x + dx, 0, side - 1)
                    np.testing.assert_array_equal(probabilities.argmax(axis=1), target.ravel())
                    self.assertEqual(np.count_nonzero(probabilities), side * side)

    def test_canonical_ties_and_walls(self):
        cases = [
            (["S  ", "   ", "  G"], ["S  ", "o  ", "ooG"]),
            (["  G", "   ", "S  "], ["  G", "  o", "Soo"]),
            (["#####", "# S #", "# # #", "# G #", "#####"],
             ["#####", "#oS #", "#o# #", "#oG #", "#####"]),
            (["#####", "#   #", "#S#G#", "#   #", "#####"],
             ["#####", "#   #", "#S#G#", "#ooo#", "#####"]),
            (["##S ", " G##", "####", "####"], ["##S ", " G##", "####", "####"]),
        ]
        for rows, expected in cases:
            with self.subTest(rows=rows):
                weights = build_weights(len(rows))
                ids = ids_for(weights, rows)
                for rounds in (2 * ids.size, 4 * ids.size):
                    self.assertEqual(output_for(weights, ids, rounds), "".join(expected))


if __name__ == "__main__":
    unittest.main()
