"""Freeze analytic classifiers on CPU and run the locked size sweep on Kaggle."""
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from frame import Transformer, inspect, rounds


ROOT = Path(__file__).resolve().parent
TASKS = ("graph","parity","chains")
DEV_COUNT = 16
BATCH_SIZE = 32


def write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded_example(x):
    return np.asarray(x.shape,dtype="<i8").tobytes() + np.asarray(x,dtype="<i8").tobytes()


def examples(task,mod,size,dev=False):
    root_seed = 41 if dev else 42
    stream = [root_seed, TASKS.index(task), size]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(stream)))
    used = set()
    if not dev and size in mod.DEV_SIZES:
        earlier, _ = examples(task,mod,size,True)
        used = {hashlib.sha256(encoded_example(x)).digest() for x,_,_ in earlier}
    count = DEV_COUNT if dev else mod.COUNT
    corpus, sha = [],hashlib.sha256()
    for index in range(count):
        for _ in range(10000):
            x,y,meta = mod.sample(size,index,rng)
            if x.dtype != np.int64 or x.ndim!=2 or not len(x):
                raise ValueError("Expected a nonempty int64 token-by-field array")
            raw = encoded_example(x)
            item = hashlib.sha256(raw).digest()
            if item not in used:
                break
        else:
            raise RuntimeError("Unable to draw distinct inputs")
        used.add(item)
        sha.update(item)
        sha.update(int(y).to_bytes(8,"little",signed=True))
        corpus.append((x,int(y),meta))
    return corpus,{"seed_sequence":stream,"count":count,"sha256":sha.hexdigest()}


def load(task):
    folder = ROOT/"models"/task
    spec = read(folder/"network.json")
    with np.load(folder/"weights.npz",allow_pickle=False) as z:
        w = {k:z[k] for k in z.files}
    return spec,w,inspect(spec,w)


def score(model,corpus,task,extra=True):
    lengths = [len(x) for x,_,_ in corpus]
    groups = defaultdict(list)
    for i,(x,_,meta) in enumerate(corpus):
        bucket = str(meta["distance"]) if task=="graph" else "all"
        groups[(bucket,len(x))].append(i)
    mults = (1,2,4) if extra else (1,)
    predictions = {m:np.full(len(corpus),-1,dtype=np.int64) for m in mults}
    by_distance = {}
    base_time = extended_time = 0.0
    batch_count = 0
    for (bucket,_), ids in groups.items():
        subset = by_distance.setdefault(bucket, {"correct":0,"count":0,"inference_seconds":0.0})
        for start in range(0,len(ids),BATCH_SIZE):
            selected = ids[start:start+BATCH_SIZE]
            x = [corpus[i][0] for i in selected]
            answers,times = model.evaluate_batch(x,mults)
            for m in mults:
                predictions[m][selected] = answers[m]
            correct = sum(int(answer)==corpus[i][1] for answer,i in zip(answers[1],selected))
            subset["correct"] += correct
            subset["count"] += len(selected)
            subset["inference_seconds"] += times[1]
            base_time += times[1]
            extended_time += times[mults[-1]]
            batch_count += 1
    y = np.asarray([ans for _,ans,_ in corpus])
    right = int((predictions[1]==y).sum())
    stats = {"count":len(corpus),"correct":right,"accuracy":right/len(corpus),
             "inference_seconds":base_time,"seconds_including_extra_checkpoints":extended_time,
             "token_range":[min(lengths),max(lengths)],
             "base_loop_range":[rounds(model.spec,min(lengths)),rounds(model.spec,max(lengths))],
             "batches":batch_count,
             "correct_at_loop_multipliers":{str(m):int((predictions[m]==y).sum()) for m in mults},
             "unchanged_at_all_checkpoints":int(np.logical_and.reduce([p==predictions[1] for p in predictions.values()]).sum())}
    if task=="graph":
        stats["by_distance"] = by_distance
    return stats,hashlib.sha256(predictions[1].astype("<i8").tobytes()).hexdigest()


def prepare(task):
    mod = importlib.import_module(task)
    folder = ROOT/"models"/task
    folder.mkdir(parents=True,exist_ok=True)
    spec,w = mod.build()
    write(folder/"network.json",spec)
    np.savez_compressed(folder/"weights.npz",**w)
    spec,w,info = load(task)
    model = Transformer(spec,w)
    rows = {}
    for size in mod.DEV_SIZES:
        data,ident = examples(task,mod,size,True)
        stats,pred_hash = score(model,data,task)
        rows[str(size)] = {"hash":pred_hash,"data":ident,"score":stats}
    record = {**info,"dev":rows,"generator_sha256":hashlib.sha256((ROOT/f"{task}.py").read_bytes()).hexdigest()}
    write(folder/"freeze.json",record)
    print(task,info,[(s,r["score"]["correct"],r["score"]["count"]) for s,r in rows.items()],flush=True)


def lock():
    result = {}
    for task in TASKS:
        mod = importlib.import_module(task)
        spec,w,info = load(task)
        frozen = read(ROOT/"models"/task/"freeze.json")
        assert info["model_sha256"]==frozen["model_sha256"]
        result[task]={str(s):examples(task,mod,s)[1] for s in mod.SIZES}
    write(ROOT/"corpus-lock.json",result)


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA device is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    locked = read(ROOT/"corpus-lock.json")
    models = {}
    with torch.inference_mode():
        for task in TASKS:
            mod = importlib.import_module(task)
            frozen = read(ROOT/"models"/task/"freeze.json")
            spec,w,info = load(task)
            if info["model_sha256"]!=frozen["model_sha256"] or hashlib.sha256((ROOT/f"{task}.py").read_bytes()).hexdigest()!=frozen["generator_sha256"]:
                raise ValueError(f"{task} freeze mismatch")
            model = Transformer(spec,w,gpu=True)
            for size in mod.DEV_SIZES:
                data,ident = examples(task,mod,size,True)
                stats,pred_hash = score(model,data,task,extra=False)
                expected = frozen["dev"][str(size)]
                if pred_hash!=expected["hash"] or ident!=expected["data"]:
                    raise ValueError(f"{task} CPU/GPU DEV mismatch at {size}")
            models[task] = model
            print(f"{task}: verified frozen tensors and CPU/GPU validation",flush=True)
        for task,model in models.items():
            mod = importlib.import_module(task)
            results = []
            for size in mod.SIZES:
                data,ident = examples(task,mod,size)
                if ident!=locked[task][str(size)]:
                    raise ValueError("The held-out corpus did not match its lock")
                stats,_ = score(model,data,task)
                results.append({"size":size,"range":"within" if size<=max(mod.DEV_SIZES) else "beyond",
                                "corpus":ident,**stats})
                print(f"{task} {size}: {stats['correct']}/{stats['count']} in {stats['inference_seconds']:.4f}s",flush=True)
            frozen = read(ROOT/"models"/task/"freeze.json")
            write(Path.cwd()/f"length-{task}.json", {"task":task,"model_sha256":frozen["model_sha256"],
                  "parameters":frozen["parameters"],"dev_sizes":mod.DEV_SIZES,"rows":results,
                  "gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"cuda":torch.version.cuda,
                  "numpy":np.__version__,"dtype":"float64","batch_size_max":BATCH_SIZE,"cpu_gpu_dev_parity":True})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","lock","gpu"))
    p.add_argument("tasks",nargs="*")
    args=p.parse_args()
    if args.mode=="prepare":
        for task in args.tasks or TASKS:
            prepare(task)
    elif args.mode=="lock":
        lock()
    else:
        gpu()
