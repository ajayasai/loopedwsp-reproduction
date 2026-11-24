"""Independently replay Kaggle's archived move IDs using bitsets for the boards."""
import argparse
from pathlib import Path

import numpy as np

from experiment import RANGES, digest, load, read, save


def hanoi_bits(n, ids, alphabet):
    full=(1<<n)-1
    pegs=[full,0,0]
    for step, token in enumerate(ids,1):
        disk,src,dst=alphabet[token]
        bit=1<<(disk-1)
        if not (0<=src<3 and 0<=dst<3 and src!=dst and 1<=disk<=n and
                pegs[src]&-pegs[src]==bit and (pegs[dst] & (bit-1))==0):
            raise ValueError(f"Hanoi {n}: illegal move {step}")
        pegs[src]^=bit
        pegs[dst]|=bit
    return pegs==[0,0,full] and len(ids)==full


def checker_bits(n, ids, alphabet):
    low=(1<<n)-1
    high=low<<(n+1)
    red,blue,empty=low,high,n
    for step,token in enumerate(ids,1):
        src,dst=alphabet[token]
        delta=dst-src
        if not (0<=src<=2*n and dst==empty and abs(delta) in (1,2)):
            raise ValueError(f"Checkers {n}: illegal destination or displacement at {step}")
        frm,to=1<<src,1<<dst
        active,opp=(red,blue) if delta>0 else (blue,red)
        if not active&frm or (abs(delta)==2 and not (opp & (1<<((src+dst)//2)))):
            raise ValueError(f"Checkers {n}: wrong color or invalid jump at {step}")
        active^=frm|to
        if delta>0:
            red=active
        else:
            blue=active
        empty=src
    return (red,blue,empty)==(high,low,n) and len(ids)==n*(n+2)


def audit(source):
    audits={}
    for task,checker in (("hanoi",hanoi_bits),("checkers",checker_bits)):
        spec,_,info=load(task)
        report=read(source/f"puzzle-{task}.json")
        if report["model"]["model_sha256"]!=info["model_sha256"]:
            raise ValueError("Model fingerprint mismatch")
        rows=report["size_rows"]
        expected=list(range(1,RANGES[task][1]+1))
        if [row["size"] for row in rows]!=expected:
            raise ValueError("Missing, duplicate or out-of-order sizes")
        verified=0
        with np.load(source/f"emitted-{task}.npz",allow_pickle=False) as archive:
            if set(archive.files)!={f"n{n}" for n in expected}:
                raise ValueError("Unexpected archive members")
            for row in rows:
                n=row["size"]
                ids=archive[f"n{n}"]
                if (ids.dtype!=np.int32 or ids.ndim!=1 or ids.min()<1 or ids.max()>=len(spec["alphabet"])
                        or digest(ids)!=row["move_ids_sha256"] or len(ids)!=row["moves"]):
                    raise ValueError(f"{task}/{n}: move archive mismatch")
                if not checker(n,ids,spec["alphabet"]):
                    raise ValueError(f"{task}/{n}: final bitset or optimal move count mismatch")
                if not (row["ended_with_eos"] and row["valid"] and row["optimal"] and
                        row["predictions_changed_at_2R_or_4R"]==0 and row["legal_moves"]==len(ids)):
                    raise ValueError(f"{task}/{n}: disagreement with remote evidence")
                verified+=len(ids)
        audits[task]={"sizes_replayed":len(rows),"moves_replayed":verified,"all_legal_goal_and_optimal":True}
    return audits


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=audit(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,result)
    print(result)
