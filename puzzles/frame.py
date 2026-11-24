"""FP64 causal residual transformer, with a checked fast path for prompt-only attention."""
import hashlib
import json
import math
import time

import numpy as np


def inspect(spec, w):
    d = w["readout"].shape[0]
    alphabet = spec["alphabet"]
    if alphabet[0] is not None or len({tuple(a) for a in alphabet[1:]}) != len(alphabet)-1:
        raise ValueError("Expected EOS followed by unique move tokens")
    if not isinstance(spec["rounds"],int) or spec["rounds"] < 1:
        raise ValueError("Invalid tied-block count")
    required = {"readout": (d,len(alphabet)), "bias": (len(alphabet),)}
    for i,f in enumerate(spec["fields"]):
        kind = f["kind"]
        sizes = {"role":2,"size":w[f"e{i}"].shape[0],"position_digit":f.get("base"),
                 "position":f.get("modulus")}
        required[f"e{i}"] = (sizes[kind],d)
    mutable = np.zeros(d,dtype=bool)
    for i,heads in enumerate(spec["layers"]):
        p = f"l{i}."
        ff = w[p+"w1"].shape[1]
        required.update({p+"w1":(d,ff),p+"w2":(ff,d),p+"b1":(ff,),p+"b2":(d,)})
        mutable |= (w[p+"w2"] != 0).any(axis=0) | (w[p+"b2"] != 0)
        for j in range(heads):
            h = p+f"h{j}."
            k,v = w[h+"q"].shape[1],w[h+"v"].shape[1]
            required.update({h+"q":(d,k),h+"k":(d,k),h+"v":(d,v),h+"o":(v,d)})
            mutable |= (w[h+"o"] != 0).any(axis=0)
    if set(required)!=set(w):
        raise ValueError(f"Unexpected tensor names: {set(required)^set(w)}")
    digest = hashlib.sha256(json.dumps(spec,sort_keys=True).encode())
    for name,shape in sorted(required.items()):
        a=w[name]
        if a.dtype!=np.float64 or a.shape!=shape or not np.isfinite(a).all():
            raise ValueError(f"Invalid tensor {name}")
        digest.update(json.dumps([name,shape,a.dtype.str]).encode())
        digest.update(np.ascontiguousarray(a).tobytes())

    # Conditions for evaluating positions independently: the schema has no field
    # for a generated symbol's identity, and all heads select only the prompt.
    margins=[]
    base=np.zeros((2,d),dtype=np.float64)
    for i,f in enumerate(spec["fields"]):
        table=w[f"e{i}"]
        base += table if f["kind"]=="role" else table[0]
    for i,heads in enumerate(spec["layers"]):
        for j in range(heads):
            p=f"l{i}.h{j}."
            q,k = w[p+"q"],w[p+"k"]
            used = (q!=0).any(axis=1) | (k!=0).any(axis=1)
            if (mutable&used).any():
                raise ValueError("Routing coordinates can change during recurrence")
            for f_id,f in enumerate(spec["fields"]):
                a=w[f"e{f_id}"][:,used]
                if f["kind"]!="role" and not np.all(a==a[:1]):
                    raise ValueError("Routing depends on the size or position")
            scores = ((base@q)@(base@k).T)/math.sqrt(q.shape[1])
            margin=float(scores[1,0]-scores[1,1])
            if not np.isfinite(margin) or margin<=750:
                raise ValueError("FP64 causal attention does not provably select the prompt")
            margins.append(margin)
    if not margins:
        raise ValueError("Missing prompt attention")
    return {"model_sha256":digest.hexdigest(),"parameters":sum(a.size for a in w.values()),
            "nonzero_parameters":sum(int(np.count_nonzero(a)) for a in w.values()),
            "prompt_logit_margins":margins,"width":d,"rounds":spec["rounds"]}


class CausalModel:
    def __init__(self,spec,w,gpu=False):
        self.info=inspect(spec,w)
        self.spec,self.gpu=spec,gpu
        if gpu:
            import torch
            self.torch=torch
            self.w={k:torch.as_tensor(v,device="cuda") for k,v in w.items()}
        else:
            self.w=w

    def embed(self,n,positions):
        h=None
        role=(positions!=0)
        role=role.long() if self.gpu else role.astype(np.int64)
        for i,f in enumerate(self.spec["fields"]):
            kind=f["kind"]
            if kind=="role":
                ids=role
            elif kind=="size":
                ids=(1-role)*n
            elif kind=="position_digit":
                ids=((positions+f.get("offset",0))//f["divisor"])%f["base"]
            else:
                ids=positions%f["modulus"]
            term=self.w[f"e{i}"][ids]
            h=term if h is None else h+term
        return h

    def block(self,h,literal=False):
        for i,heads in enumerate(self.spec["layers"]):
            p=f"l{i}."
            after=h
            for j in range(heads):
                k=p+f"h{j}."
                if literal:
                    q,key=h@self.w[k+"q"],h@self.w[k+"k"]
                    scores=(q@key.T)/math.sqrt(key.shape[1])
                    if self.gpu:
                        mask=self.torch.ones_like(scores,dtype=self.torch.bool).triu(1)
                        probs=scores.masked_fill(mask,-float("inf")).softmax(dim=-1)
                    else:
                        scores[np.triu_indices(len(h),1)]=-np.inf
                        e=np.exp(scores-scores.max(axis=1,keepdims=True))
                        probs=e/e.sum(axis=1,keepdims=True)
                    v=probs@(h@self.w[k+"v"])
                else:
                    v=h[:1]@self.w[k+"v"]
                after=after + v@self.w[k+"o"]
            z=after@self.w[p+"w1"]+self.w[p+"b1"]
            act=self.torch.relu(z) if self.gpu else np.maximum(z,0)
            h=after+act@self.w[p+"w2"]+self.w[p+"b2"]
        return h

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    def token_ids(self,h):
        ids=(h@self.w["readout"]+self.w["bias"]).argmax(axis=-1)
        return ids.cpu().numpy() if self.gpu else ids

    def literal_decode(self,n,limit=256):
        """Greedy, one-token-at-a-time reference, rerunning the complete causal prefix."""
        emitted=[]
        for step in range(limit):
            positions=self.torch.arange(step+1,device="cuda") if self.gpu else np.arange(step+1)
            h=self.embed(n,positions)
            # By the model's field schema, each appended move contributes the
            # same role embedding and receives its actual position in this prefix.
            for _ in range(self.spec["rounds"]):
                h=self.block(h,literal=True)
            token=int(self.token_ids(h[-1:])[0])
            if token==0:
                return np.asarray(emitted,dtype=np.int32)
            emitted.append(token)
        raise RuntimeError("Literal decoder reached the DEV guard before EOS")

    def decode(self,n,limit=2**21,max_chunk=4096):
        """Emit through the first model EOS; the guard and windows are independent of n."""
        emitted=[]
        pos=0
        base_seconds=0.0
        changed_positions=0
        evaluated=0
        stop=False
        self.sync()
        wall=time.perf_counter()
        while pos<limit:
            start=time.perf_counter()
            count=min(max(16,pos),max_chunk,limit-pos)
            # Row zero is a copy of the actual causal prompt for this window.
            indices=np.empty(count+1,dtype=np.int64)
            indices[0]=0
            indices[1:]=np.arange(pos,pos+count)
            indices=self.torch.as_tensor(indices,device="cuda") if self.gpu else indices
            h=self.embed(n,indices)
            rounds=self.spec["rounds"]
            for _ in range(rounds):
                h=self.block(h)
            ids=self.token_ids(h[1:]).astype(np.int32)
            eos=np.flatnonzero(ids==0)
            end=int(eos[0]) if len(eos) else count
            stop=end<count
            emitted.append(ids[:end].copy())
            base_seconds+=time.perf_counter()-start
            valid_positions=end+int(stop)
            changed=np.zeros(valid_positions,dtype=bool)
            t=rounds
            for target in (2*rounds,4*rounds):
                while t<target:
                    new=self.block(h)
                    t+=1
                    fixed=self.torch.equal(new,h) if self.gpu else np.array_equal(new,h)
                    h=new
                    if fixed:
                        t=4*rounds
                        break
                other=self.token_ids(h[1:valid_positions+1])
                changed |= other!=ids[:valid_positions]
            changed_positions+=int(changed.sum())
            pos+=count
            evaluated+=count
            if stop:
                break
        start=time.perf_counter()
        ids=np.concatenate(emitted)
        base_seconds+=time.perf_counter()-start
        self.sync()
        return ids, {"ended_with_eos":stop,"decode_seconds":base_seconds,
                     "decode_and_stability_seconds":time.perf_counter()-wall,
                     "positions_evaluated_in_windows":evaluated,
                     "decoding_steps":len(ids)+int(stop),
                     "predictions_changed_at_2R_or_4R":changed_positions}
