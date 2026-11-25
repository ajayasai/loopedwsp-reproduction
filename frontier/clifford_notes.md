# Predicting measurements of an ideal Clifford circuit

Prepare three qubits in `|000⟩`, apply a sequence of gates, and ask which measurements have a predictable outcome. Each observable is a three-letter word in qubit order 0, 1, 2. For example, `ZZI` multiplies the Z measurements on the first two qubits and ignores the third. An expectation of +1 or −1 predicts that product with certainty. An expectation of 0 gives equal probabilities for +1 and −1. Each query describes a separate preparation of the same ideal state.

From the repository root, run the Bell-pair example.

```sh
python frontier/experiment.py predict clifford --input frontier/examples/clifford.json
```

The first gate, `["H",0]`, creates a superposition on qubit 0. The second, `["CX",0,1]`, couples qubit 1 to it. The measurements `XXI, YYI, ZZI, ZII, IZI, IIZ` return `{"expectations":[1,-1,1,0,0,1]}`. Each member of the pair has an unpredictable Z measurement, but the product of their Z outcomes is always +1. Their Y outcomes always have product −1. Reversing those two gates gives the same measurement list expectations of `[0,0,0,0,1,1]`.

## Circuit and model

Supply up to 64 gates in chronological order and 1–64 Pauli queries. Single-qubit gates are `["H",q]`, `["S",q]` and `["Sdg",q]`. Sdg denotes the inverse of S. A controlled-NOT gate takes a distinct control and target, as in `["CX",0,2]`. The model assumes ideal gates, the specified initial state and Pauli measurements.

The frozen attention weights encode how each gate transforms a single Pauli word backward through the circuit. At each step the current word addresses a table that returns a new word and its sign. The model accumulates these signs, then reads the initial state's expectation for the final word. The 1,025 FP32 coefficients comprise 960 signed transitions, 64 initial-state values and one shared attention scale. The model reuses the same weights at each gate. The separate reference evolves eight complex amplitudes forward and takes inner products with the requested observables. [Gottesman’s Heisenberg representation](https://arxiv.org/abs/quant-ph/9807006)

Half the generator modes sample arbitrary gate sequences. The other two modes prepare a Bell pair or a three-qubit GHZ state with a relative phase of +i or −i, then insert adjacent inverse-gate pairs to reach the requested length. For each circuit the generator uses the reference to sample four deterministic and four unbiased nonidentity observables and shuffles them. The circuit and unsigned observable words are the entire model input. The benchmark measures exact expectation classification on synthetic circuits of three qubits.
