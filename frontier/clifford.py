"""Program signed Heisenberg propagation for Pauli queries on three-qubit circuits."""

from itertools import product

import numpy as np
import torch
from torch import nn


ALPHABET = "IXYZ"
WORDS = tuple("".join(p) for p in product(ALPHABET, repeat=3))
GATES = tuple((g, q) for g in ("H", "S", "Sdg") for q in range(3)) + tuple(
    ("CX", control, target) for control in range(3) for target in range(3) if control != target
)
GATE_ID = {gate: i for i, gate in enumerate(GATES)}
WORD_ID = {word: i for i, word in enumerate(WORDS)}
MAX_GATES = 64


def encode(record):
    """Each query row stores its literal Pauli-word ID followed by the common gate IDs."""
    if not isinstance(record, dict) or set(record) != {"gates", "observables"}:
        raise ValueError("Supply gates and observables")
    gates, words = record["gates"], record["observables"]
    if not isinstance(gates, list) or len(gates) > MAX_GATES:
        raise ValueError("Supply up to 64 gates")
    if (not isinstance(words, list) or not 1 <= len(words) <= 64
            or any(not isinstance(w, str) or w not in WORD_ID for w in words)):
        raise ValueError("Supply 1 to 64 Pauli words, each three letters drawn from I, X, Y and Z")
    ids = []
    for gate in gates:
        if (not isinstance(gate, list) or not gate or not isinstance(gate[0], str)
                or any(type(q) is not int for q in gate[1:]) or tuple(gate) not in GATE_ID):
            raise ValueError("Use H, S or Sdg on qubit 0–2, or CX with distinct control and target")
        ids.append(GATE_ID[tuple(gate)])
    return np.array([[WORD_ID[w], *ids] for w in words], dtype=np.int64)


def _conjugate(gate, word):
    """Return sign and letters for U† P U using local Clifford algebra."""
    letters = list(word)
    name, q = gate[:2]
    if name == "CX":
        t = gate[2]
        xc, zc = letters[q] in "XY", letters[q] in "YZ"
        xt, zt = letters[t] in "XY", letters[t] in "YZ"
        # A sign arises when the control has X and the target has Z
        # and their other two binary Pauli coordinates agree.
        minus = xc and zt and (xt == zc)
        pauli = ("I", "Z", "X", "Y")
        letters[q] = pauli[2*xc + (zc ^ zt)]
        letters[t] = pauli[2*(xt ^ xc) + zt]
    else:
        substitutions = {"H": ("IZYX", "Y"), "S": ("IYXZ", "X"), "Sdg": ("IYXZ", "Y")}
        replacements, negative = substitutions[name]
        minus = letters[q] == negative
        letters[q] = replacements[ALPHABET.index(letters[q])]
    return (-1 if minus else 1), "".join(letters)


class PauliTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # A signed one-based code represents the output word and its phase.
        memory = []
        for gate in GATES:
            row = []
            for word in WORDS:
                sign, out = _conjugate(gate, word)
                row.append(sign*(1+WORD_ID[out]))
            memory.append(row)
        self.transitions = nn.Parameter(torch.tensor(memory, dtype=torch.float32), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(128.), requires_grad=False)
        initial = [float(all(p in "IZ" for p in word)) for word in WORDS]
        self.vacuum = nn.Parameter(torch.tensor(initial), requires_grad=False)

    def address(self, ids, slots):
        # Dot products encode minus squared address distance up to a query constant.
        keys = torch.arange(slots, dtype=ids.dtype, device=ids.device)
        queries = torch.stack((2*ids, torch.ones_like(ids)), dim=-1)
        dots = queries @ torch.stack((keys, -keys.square()))
        return (self.sharpness*dots).softmax(dim=-1)

    def forward(self, tokens):
        if tokens.ndim != 3 or not 1 <= tokens.shape[2] <= 1+MAX_GATES:
            raise ValueError("Expected batch × queries × (one word plus gate IDs)")
        raw = tokens.to(self.transitions.dtype)
        words = raw[:, :, 0]
        signs = torch.ones_like(words)
        # Every query in a record refers to the same circuit. Gate attention
        # reads the required transfer memories in reverse chronological order.
        gates = self.address(raw[:, 0, 1:].flip(-1), len(GATES)) @ self.transitions
        for gate in gates.unbind(1):
            signed = (self.address(words, len(WORDS)) @ gate.unsqueeze(-1)).squeeze(-1)
            signs = signs * signed.clamp(-1, 1)
            words = signed.abs() - 1
        expected = signs * (self.address(words, len(WORDS)) @ self.vacuum)
        return torch.stack((-(expected+1).abs(), -expected.abs(), -(expected-1).abs()), dim=-1)


def build_model():
    return PauliTransformer().eval()


# An independent Schrödinger-picture reference evolves eight complex amplitudes.
PAULI = np.array([[[1,0],[0,1]], [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
OBSERVABLES = np.stack([np.kron(np.kron(a,b),c) for a,b,c in product(PAULI, repeat=3)])
SINGLE = {"H": np.array([[1,1],[1,-1]]) / np.sqrt(2),
          "S": np.diag([1,1j]), "Sdg": np.diag([1,-1j])}


def _expectations(gates):
    amplitudes = np.zeros(8, dtype=np.complex128)
    amplitudes[0] = 1
    for name, *qubits in gates:
        if name == "CX":
            c, t = qubits
            # CX permutes computational-basis states and is its own inverse.
            indices = np.arange(8)
            amplitudes = amplitudes[indices ^ (((indices >> (2-c)) & 1) << (2-t))]
        else:
            q = qubits[0]
            vector = np.moveaxis(amplitudes.reshape(2,2,2), q, 0).reshape(2,4)
            transformed = (SINGLE[name] @ vector).reshape(2,2,2)
            amplitudes = np.moveaxis(transformed, 0, q).reshape(8)
    products = np.einsum("i,pij,j->p", amplitudes.conj(), OBSERVABLES, amplitudes)
    exact = np.rint(products.real).astype(np.int64)
    if not np.allclose(products, exact, rtol=0, atol=1e-10):
        raise ArithmeticError("Unexpected non-Clifford expectation")
    return exact


def reference(record):
    encode(record)
    all_values = _expectations(record["gates"])
    return np.array([1+all_values[WORD_ID[w]] for w in record["observables"]], dtype=np.int64)


def generate(rng, size, index):
    if size < 4 or size > MAX_GATES or size % 2:
        raise ValueError("Benchmark lengths must be even and between 4 and 64")
    if index % 4 < 2:
        gates = [GATES[int(i)] for i in rng.integers(len(GATES), size=size)]
    else:
        a, b, c = map(int, rng.permutation(3))
        gates = [("H",a), ("CX",a,b)]
        if index % 4 == 3:
            gates.extend([("CX",b,c), (str(rng.choice(["S","Sdg"])),a)])
        while len(gates) < size:
            gate = GATES[int(rng.integers(len(GATES)))]
            inverse = ({"S":"Sdg", "Sdg":"S"}.get(gate[0], gate[0]), *gate[1:])
            where = int(rng.integers(len(gates)+1))
            gates[where:where] = [gate, inverse]
    # Balance deterministic and unbiased readouts. Only the unsigned queries
    # and literal circuit reach the model. The identity is omitted from this set.
    values = _expectations(gates)
    deterministic = np.flatnonzero(values[1:]) + 1
    random = np.flatnonzero(values == 0)
    ids = np.concatenate((rng.choice(deterministic, size=4, replace=False), rng.choice(random, size=4, replace=False)))
    return {"gates":[list(g) for g in gates], "observables":[WORDS[i] for i in rng.permutation(ids)]}


def example():
    return {"gates":[["H",0],["CX",0,1]], "observables":["XXI","YYI","ZZI","ZII","IZI","IIZ"]}


def format_output(record, prediction):
    if len(prediction) != len(record["observables"]) or any(label not in (0,1,2) for label in prediction):
        raise ValueError("Expected one expectation class for each observable")
    return {"expectations":[int(label)-1 for label in prediction]}
