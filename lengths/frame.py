"""FP64 residual softmax/ReLU transformer for integer token fields; last-token readout."""
import hashlib
import json
import math
import time

import numpy as np


def rounds(spec, n):
    return {"tokens": n, "tokens_plus_one": n+1,
            "log2_plus_one": (n-1).bit_length()+1}[spec["round_rule"]]


def inspect(spec, w):
    d = w["e0"].shape[1]
    required = {f"e{i}": (w[f"e{i}"].shape[0],d) for i in range(len(spec["fields"]))}
    classes = len(w["bias"])
    required.update(readout=(d,classes),bias=(classes,))
    for i, heads in enumerate(spec["layers"]):
        p = f"l{i}."
        ff = w[p+"w1"].shape[1]
        required.update({p+"w1": (d,ff),p+"b1": (ff,),p+"w2":(ff,d),p+"b2":(d,)})
        for j in range(heads):
            key = p+f"h{j}."
            k, v = w[key+"q"].shape[1],w[key+"v"].shape[1]
            required.update({key+"q":(d,k),key+"k":(d,k),key+"v":(d,v),key+"o":(v,d)})
    if set(w) != set(required):
        raise ValueError(f"Unexpected model tensors: {set(w)^set(required)}")
    sha = hashlib.sha256(json.dumps(spec,sort_keys=True).encode())
    for name, shape in sorted(required.items()):
        a = w[name]
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError(f"Invalid FP64 tensor {name}")
        sha.update(json.dumps([name,shape,a.dtype.str]).encode())
        sha.update(np.ascontiguousarray(a).tobytes())
    return {"model_sha256":sha.hexdigest(),"parameters":sum(a.size for a in w.values()),"width":d}


class Transformer:
    def __init__(self, spec, arrays, gpu=False, literal=False):
        self.gpu, self.literal, self.spec = gpu, literal, spec
        if gpu:
            import torch
            self.torch = torch
            self.array = lambda a: torch.as_tensor(a, device="cuda")
        else:
            self.array = np.asarray
        self.w = {k:self.array(v) for k,v in arrays.items()}
        changes = np.zeros(arrays["e0"].shape[1], np.bool_)
        for i, heads in enumerate(spec["layers"]):
            p = f"l{i}."
            changes |= (arrays[p+"w2"] != 0).any(axis=0) | (arrays[p+"b2"] != 0)
            for j in range(heads):
                changes |= (arrays[p+f"h{j}.o"] != 0).any(axis=0)
        self.layers = []
        for i, heads in enumerate(spec["layers"]):
            p = f"l{i}."
            layer = []
            for j in range(heads):
                key = p+f"h{j}."
                static = not (arrays[key+"q"][changes].any() or arrays[key+"k"][changes].any())
                layer.append((key, static and not literal))
            self.layers.append((p,layer))

    def equal(self, a, b):
        return self.torch.equal(a,b) if self.gpu else np.array_equal(a,b)

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    def routes(self, h, key):
        q,k = h@self.w[key+"q"],h@self.w[key+"k"]
        b,n,dk = q.shape
        if b>1 and not self.literal:
            if self.gpu:
                shared = self.equal(q,q[:1].expand_as(q)) and self.equal(k,k[:1].expand_as(k))
            else:
                shared = np.all(q == q[:1]) and np.all(k == k[:1])
            if shared:
                q,k = q[:1],k[:1]
        tiles = []
        # Split by query rows only. Every softmax row still sees every key.
        chunk = max(1,min(n, 4_000_000//(len(q)*n)))
        for i in range(0,n,chunk):
            scores = (q[:,i:i+chunk] @ k.swapaxes(-1,-2)) / math.sqrt(dk)
            if self.gpu:
                weights = scores.softmax(dim=-1)
            else:
                exps = np.exp(scores-scores.max(axis=-1,keepdims=True))
                weights = exps/exps.sum(axis=-1,keepdims=True)
            one_hot = False
            if not self.literal:
                rows = weights.shape[0]*weights.shape[1]
                one_hot = (weights!=0).sum().item()==rows and (weights==1).sum().item()==rows
            tiles.append((one_hot,weights.argmax(axis=-1) if one_hot else weights))
        return tiles

    def attend(self, tiles, v):
        pieces = []
        for single, obj in tiles:
            if single and self.gpu:
                ids = obj.unsqueeze(-1).expand(v.shape[0],-1,v.shape[2])
                piece = v.gather(1,ids)
            elif single:
                piece = v[np.arange(len(v))[:,None],obj]
            else:
                piece = obj@v
            pieces.append(piece)
        if len(pieces)==1:
            return pieces[0]
        return self.torch.cat(pieces,dim=1) if self.gpu else np.concatenate(pieces,axis=1)

    def step(self,h,cache):
        for p,heads in self.layers:
            after = h
            for key,static in heads:
                tiles = cache.get(key)
                if tiles is None:
                    tiles = self.routes(h,key)
                    if static:
                        cache[key] = tiles
                after = after + self.attend(tiles,h@self.w[key+"v"]) @ self.w[key+"o"]
            z = after@self.w[p+"w1"] + self.w[p+"b1"]
            act = self.torch.relu(z) if self.gpu else np.maximum(z,0)
            h = after + act@self.w[p+"w2"] + self.w[p+"b2"]
        return h

    def evaluate_batch(self, arrays, multiples=(1,2,4), per_token=False):
        self.sync()
        start = time.perf_counter()
        ids = self.array(np.stack(arrays))
        h = self.w["e0"][ids[:,:,0]]
        for i in range(1,ids.shape[2]):
            h = h + self.w[f"e{i}"][ids[:,:,i]]
        budget = rounds(self.spec,ids.shape[1])
        answers,timings,cache = {},{},{}
        t, fixed = 0,False
        for mult in multiples:
            while t < mult*budget and not fixed:
                updated = self.step(h,cache)
                t += 1
                if t % 4 == 0:
                    fixed = self.equal(h,updated)
                h = updated
            state = h if per_token else h[:,-1]
            logits = state@self.w["readout"] + self.w["bias"]
            pred = logits.argmax(axis=-1)
            answers[mult] = pred.cpu().numpy() if self.gpu else pred.copy()
            self.sync()
            timings[mult] = time.perf_counter()-start
        return answers,timings
