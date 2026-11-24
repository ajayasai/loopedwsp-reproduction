"""Compare cached GPU attention with literal dense CPU attention on synthetic tokens."""
import json

import numpy as np
import torch

from looped import TorchBlock, dense_numpy
from weights import build_weights


def main():
    w = build_weights(5)
    ids = np.random.default_rng(23).integers(len(w["chars"]), size=(3, 25))
    tested = (0, 1, 2, 4, 8, 16, 32)
    with torch.inference_mode():
        compiled = TorchBlock(w, torch.float64, torch.device("cuda"))
        state = compiled.token[torch.as_tensor(ids, device="cuda")] + compiled.pos
        for t in range(max(tested) + 1):
            if t in tested:
                dense = dense_numpy(w, ids, t)
                actual = (state @ compiled.readout + compiled.bias).cpu().numpy()
                np.testing.assert_array_equal(actual, dense)
            if t < max(tested):
                state = compiled.step(state)
    print(json.dumps({"backend_check": "passed", "synthetic_grids": len(ids),
                      "dense_cpu_vs_gpu_logits_identical_at": tested,
                      "device": torch.cuda.get_device_name(), "heads": compiled.head_types}))


if __name__ == "__main__":
    main()
