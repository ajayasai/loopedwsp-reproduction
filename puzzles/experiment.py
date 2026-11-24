"""Freeze the puzzle models on CPU and evaluate the complete size sweep on Kaggle."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"

import numpy as np
from frame import CausalModel, inspect
from verify import verify_checkers, verify_hanoi


HERE=Path(__file__).resolve().parent
RANGES={"hanoi":(6,20),"checkers":(3,40)}
GRADERS={"hanoi":verify_hanoi,"checkers":verify_checkers}


def save(path,d):
    path.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(task):
    folder=HERE/"models"/task
    spec=read(folder/"network.json")
    with np.load(folder/"weights.npz",allow_pickle=False) as z:
        weights={k:z[k] for k in z.files}
    return spec,weights,inspect(spec,weights)


def digest(ids):
    return hashlib.sha256(np.asarray(ids,dtype="<i4").tobytes()).hexdigest()


def replay(task,n,ids,alphabet):
    if len(ids) and (ids.min()<1 or ids.max()>=len(alphabet)):
        raise ValueError("Sequence contains an out-of-range move token or internal EOS")
    return GRADERS[task](n,(alphabet[i] for i in ids))


def prepare(task):
    folder=HERE/"models"/task
    folder.mkdir(parents=True,exist_ok=True)
    spec,w=importlib.import_module(task).build()
    save(folder/"network.json",spec)
    np.savez_compressed(folder/"weights.npz",**w)
    spec,w,info=load(task)
    model=CausalModel(spec,w)
    results={}
    for n in range(1,RANGES[task][0]+1):
        literal=model.literal_decode(n)
        ids,details=model.decode(n,limit=256)
        if not np.array_equal(literal,ids):
            raise ValueError(f"{task}/{n}: literal causal decoding and batched decoding disagree")
        grade=replay(task,n,ids,spec["alphabet"])
        if not grade["optimal"] or not details["ended_with_eos"] or details["predictions_changed_at_2R_or_4R"]:
            raise ValueError(f"{task}/{n} failed development: {grade}, {details}")
        results[str(n)]={"moves":len(ids),"move_ids_sha256":digest(ids),"literal_matches":True,
                         "stable":True,"legal_and_optimal":True}
    save(folder/"freeze.json",{**info,"dev":results})
    print(task,info,"verified DEV",list(results),flush=True)


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA GPU is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():
        models={}
        for task in RANGES:
            spec,w,info=load(task)
            freeze=read(HERE/"models"/task/"freeze.json")
            if info["model_sha256"]!=freeze["model_sha256"]:
                raise ValueError(f"{task} freeze mismatch")
            model=CausalModel(spec,w,gpu=True)
            for n in range(1,RANGES[task][0]+1):
                ids,details=model.decode(n,limit=256)
                if digest(ids)!=freeze["dev"][str(n)]["move_ids_sha256"] or not details["ended_with_eos"]:
                    raise ValueError(f"{task}/{n} disagrees with literal CPU decoder")
            models[task]=(model,info)
            print(f"Verified {task} on {torch.cuda.get_device_name()}",flush=True)
        for task,(model,info) in models.items():
            dev,max_n=RANGES[task]
            rows,archive=[],{}
            for n in range(1,max_n+1):
                start=time.perf_counter()
                ids,stats=model.decode(n)
                then=time.perf_counter()
                grade=replay(task,n,ids,model.spec["alphabet"])
                end=time.perf_counter()
                archive[f"n{n}"]=ids
                rows.append({"size":n,"range":"development" if n<=dev else "held out",**stats,**grade,
                             "replay_seconds":end-then,"wall_seconds_including_stability_and_replay":end-start,
                             "move_ids_sha256":digest(ids)})
                print(f"{task} {n}: {grade['legal_moves']}/{grade['moves']} legal, "
                      f"goal={grade['goal']} optimal={grade['optimal']} decode={stats['decode_seconds']:.5f}s",flush=True)
            np.savez_compressed(f"emitted-{task}.npz",**archive)
            save(Path(f"puzzle-{task}.json"),{"task":task,"model":info,"dtype":"float64",
                 "device":torch.cuda.get_device_name(),"torch":str(torch.__version__),"cuda":torch.version.cuda,
                 "cpu_literal_gpu_dev_parity":True,"max_window":4096,"size_rows":rows,
                 "emitted_archive_sha256":hashlib.sha256(Path(f"emitted-{task}.npz").read_bytes()).hexdigest()})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","gpu"])
    parser.add_argument("tasks",nargs="*")
    args=parser.parse_args()
    if args.mode=="prepare":
        for task in args.tasks or RANGES:
            prepare(task)
    else:
        gpu()
