"""The shared character tokenizer and dense FP64 transformer frame for all five caps."""
import hashlib
import json
import math

import numpy as np


def field_ids(text, field):
    kind = field["kind"]
    if kind == "char":
        vocab = field["vocab"]
        index = {ch: i for i, ch in enumerate(vocab)}
        return np.array([index.get(ch, len(vocab)) for ch in text], dtype=np.int64)
    pos = np.arange(len(text), dtype=np.int64)
    if kind == "pos_mod":
        return pos % field["modulus"]
    if kind == "pos_digit":
        return (pos ** field["power"] // field["base"] ** field["place"]) % field["base"]
    if kind == "coord_digit":
        rows, cols = [], []
        r = c = 0
        for ch in text:
            rows.append(r)
            cols.append(c)
            if ch == "\n":
                r, c = r+1, 0
            else:
                c += 1
        rows, cols = np.asarray(rows), np.asarray(cols)
        origin = field.get("origin",0)
        oy,ox = (origin,origin) if isinstance(origin,int) else origin
        coordinate = {"line": rows, "column": cols,
                      "squared_norm": (rows-oy)**2 + (cols-ox)**2}[field["coordinate"]]
        return (coordinate // field["base"] ** field["place"]) % field["base"]
    raise ValueError(f"Unknown tokenizer field {kind}")


def embed(spec, w, texts):
    h = np.zeros((len(texts), len(texts[0]), w["e0"].shape[1]), dtype=np.float64)
    for i, field in enumerate(spec["fields"]):
        if field["kind"] in ("char", "coord_digit"):
            ids = np.stack([field_ids(t, field) for t in texts])
        else:
            ids = field_ids(texts[0], field)
        h += w[f"e{i}"][ids]
    return h


def inspect(spec, w, n):
    d = w["e0"].shape[1]
    required = {}
    for i, field in enumerate(spec["fields"]):
        sizes = {"char": len(field.get("vocab", "")) + 1, "pos_digit": field.get("base"),
                 "pos_mod": field.get("modulus"), "coord_digit": field.get("base")}
        required[f"e{i}"] = (sizes[field["kind"]], d)
    abc = spec["alphabet"]
    if len(abc) != len(set(abc)) or "" not in abc:
        raise ValueError("Output alphabet must contain the empty string and be unambiguous")
    required["readout"], required["bias"] = (d, len(abc)), (len(abc),)
    # Conservative dense upper bound: Q/K/V/O, both attention matmuls, the
    # pointwise softmax/elementwise ops, both MLP matrices, residuals, readout.
    per_block = 0
    for i, heads in enumerate(spec["layers"]):
        p = f"l{i}."
        ff = w[p+"w1"].shape[1]
        required.update({p+"w1": (d, ff), p+"b1": (ff,), p+"w2": (ff,d), p+"b2": (d,)})
        per_block += n * (4*d*ff + 3*ff + 4*d)
        for j in range(heads):
            h = p+f"h{j}."
            k, v = w[h+"q"].shape[1], w[h+"v"].shape[1]
            required.update({h+"q": (d,k), h+"k": (d,k), h+"v": (d,v), h+"o": (v,d)})
            per_block += 4*n*d*(k+v) + n*n*(2*(k+v)+12) + 2*n*d
    if set(required) != set(w):
        raise ValueError(f"Unexpected model arrays: {set(w)^set(required)}")
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for name, shape in sorted(required.items()):
        a = np.ascontiguousarray(w[name])
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError(f"Invalid weight {name} with shape {a.shape}, expected {shape}")
        digest.update(json.dumps([name,shape,a.dtype.str]).encode())
        digest.update(a.tobytes())
    return {"parameters": sum(a.size for a in w.values()),
            "nonzero_parameters": sum(int(np.count_nonzero(a)) for a in w.values()),
            "model_sha256": digest.hexdigest(), "d_model": d,
            "dense_flops_upper_bound": per_block*n*spec["rounds_per_token"] + n*d*(2*len(abc)+len(spec["fields"])+10)}


class Frame:
    def __init__(self, spec, w, n, cuda=False, literal=False, template=None):
        self.spec, self.w, self.cuda = spec, w, cuda
        if cuda:
            import torch
            self.torch = torch
            self.array = lambda a: torch.as_tensor(a, device="cuda", dtype=torch.float64)
        else:
            self.array = np.asarray
        self.out, self.bias = self.array(w["readout"]), self.array(w["bias"])
        variable = np.zeros(w["readout"].shape[0], dtype=bool)
        for i, f in enumerate(spec["fields"]):
            if f["kind"] == "char":
                table = w[f"e{i}"]
                variable |= (table != table[:1]).any(axis=0)
        for i, heads in enumerate(spec["layers"]):
            for j in range(heads):
                variable |= (w[f"l{i}.h{j}.o"] != 0).any(axis=0)
            variable |= (w[f"l{i}.w2"] != 0).any(axis=0) | (w[f"l{i}.b2"] != 0)
        self.layout = None
        if any(f["kind"] == "coord_digit" for f in spec["fields"]):
            if template is None:
                raise ValueError("Coordinate tokenization needs a text layout")
            self.layout = tuple(i for i, ch in enumerate(template) if ch == "\n")
        ref = self.array(embed(spec, w, [template if template is not None else " "*n])[0])
        self.layers, self.head_types = [], []
        self.certified_one_hot_steps = 0
        for i, heads in enumerate(spec["layers"]):
            prepared = []
            for j in range(heads):
                p = f"l{i}.h{j}."
                q, k, v, o = [self.array(w[p+s]) for s in "qkvo"]
                scale = math.sqrt(q.shape[1])
                route = matrix = None
                invariant = not ((w[p+"q"][variable] != 0).any() or (w[p+"k"][variable] != 0).any())
                if invariant and not literal:
                    matrix = self.softmax((ref@q) @ (ref@k).T / scale)
                    count = (matrix != 0).sum().item()
                    if count == n and ((matrix == 1).sum().item() == n):
                        route = matrix.argmax(axis=1)
                        matrix = None
                self.head_types.append("one-hot" if route is not None else "static" if matrix is not None else "dense")
                prepared.append((q,k,v,o,scale,route,matrix))
            ff = [self.array(w[f"l{i}.{s}"]) for s in ("w1","b1","w2","b2")]
            self.layers.append((prepared,ff))

    def softmax(self, scores):
        if self.cuda:
            return scores.softmax(dim=-1)
        exps = np.exp(scores-scores.max(axis=-1, keepdims=True))
        return exps / exps.sum(axis=-1, keepdims=True)

    def step(self, h):
        for heads, (w1,b1,w2,b2) in self.layers:
            after = h
            for q,k,v,o,scale,route,matrix in heads:
                values = h @ v
                if route is not None:
                    attended = values[:,route,:]
                elif matrix is not None:
                    attended = matrix @ values
                else:
                    scores = (h@q) @ (h@k).swapaxes(-1,-2) / scale
                    if self.cuda and scores.shape[-1] >= 2:
                        # With this margin every losing exponential rounds to
                        # zero in binary64. Use the ordinary softmax otherwise.
                        pair = scores.topk(2, dim=-1)
                        gap = pair.values[...,0] - pair.values[...,1]
                        if bool((gap > 750).all().item()):
                            idx = pair.indices[...,0,None].expand(-1,-1,values.shape[-1])
                            attended = values.gather(1,idx)
                            self.certified_one_hot_steps += 1
                        else:
                            attended = self.softmax(scores) @ values
                    else:
                        attended = self.softmax(scores) @ values
                after = after + attended @ o
            x = after @ w1 + b1
            activated = self.torch.relu(x) if self.cuda else np.maximum(x,0)
            h = after + activated @ w2 + b2
        return h

    def predict(self, texts, rounds=None):
        if self.layout is not None and any(tuple(i for i,c in enumerate(t) if c == "\n") != self.layout for t in texts):
            raise ValueError("Rebuild the cached frame when newline positions change")
        h = self.array(embed(self.spec, self.w, texts))
        budget = len(texts[0])*self.spec["rounds_per_token"] if rounds is None else rounds
        for t in range(1, budget+1):
            new = self.step(h)
            if t%16 == 0:
                fixed = self.torch.equal(h,new) if self.cuda else np.array_equal(h,new)
                if fixed:
                    h = new
                    break
            h = new
        logits = h@self.out + self.bias
        ids = logits.argmax(axis=-1)
        ids = ids.cpu().numpy() if self.cuda else ids
        return ["".join(self.spec["alphabet"][i] for i in row) for row in ids], t
