"""Prepare cap artifacts, check them on CPU DEV, and score frozen files on Kaggle."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for option in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
from frame import Frame, inspect

CAPS = (1000, 5000, 10000, 20000, 40000)
HERE = Path(__file__).resolve().parent
REV = "549754de3d67eebe57b6b22c7d226b9412d55b07"
TRAIN_SHA = "25f9e39b5aee16967da41ba63b2d1d75cf608c2360d8a1419d827f9859dbafa0"
TEST_SHA = "5120d52088fb3f8036e62e90103808a79ba58847d8601f9451f40991b0903a62"
DEV_COUNT = 8


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def text_grid(flat):
    if len(flat) != 900:
        raise ValueError("Dataset requires 900 cells")
    return "\n".join(flat[i:i+30] for i in range(0,900,30))


def corpus(path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("Dataset file checksum mismatch")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1000:
        raise ValueError("Expected 1,000 official grids")
    return [(text_grid(r["question"]), text_grid(r["answer"])) for r in rows]


def answer_hash(answers):
    return hashlib.sha256(json.dumps(answers,ensure_ascii=True).encode()).hexdigest()


def load(folder):
    spec = read(folder/"network.json")
    with np.load(folder/"weights.npz",allow_pickle=False) as z:
        w = {k:z[k] for k in z.files}
    return spec, w, inspect(spec,w,929)


def prepare(cap):
    folder = HERE/f"c{cap}"
    builder = importlib.util.spec_from_file_location(f"cap{cap}", folder/"write_weights.py")
    module = importlib.util.module_from_spec(builder)
    builder.loader.exec_module(module)
    spec, w = module.build()
    save(folder/"network.json",spec)
    np.savez_compressed(folder/"weights.npz",**w)
    spec,w,checked = load(folder)
    if checked["parameters"]>cap or checked["dense_flops_upper_bound"]>10**14:
        raise ValueError(f"Over budget: {checked}")
    synthetic = ["S # \n# G "]
    for rounds in (1,4):
        cached = Frame(spec,w,len(synthetic[0]),template=synthetic[0]).predict(synthetic,rounds)[0]
        literal = Frame(spec,w,len(synthetic[0]),literal=True,template=synthetic[0]).predict(synthetic,rounds)[0]
        if cached != literal:
            raise ValueError("Cached and dense NumPy disagree on synthetic text")
    train = corpus(HERE.parent/"data/maze/train.csv",TRAIN_SHA)[:DEV_COUNT]
    ledger = folder/"dev-ledger.json"
    calls = read(ledger) if ledger.exists() else []
    if len(calls)>=20:
        raise RuntimeError("DEV allowance spent")
    calls.append({"call":len(calls)+1, "subset":f"train.csv rows 1–{DEV_COUNT}", "model":checked,
                  "status":"started"})
    save(ledger,calls)
    q,labels = zip(*train)
    reference = Frame(spec,w,929,template=q[0])
    predictions,steps = reference.predict(q)
    solved = sum(p.strip("\n")==a for p,a in zip(predictions,labels))
    calls[-1].update(status="complete", solved=solved, count=len(train), actual_blocks=steps)
    save(ledger,calls)
    record = {"cap":cap,**checked,"cpu_dev_count":len(q),"cpu_dev_exact":solved,
              "cpu_answers_sha256":answer_hash(predictions), "cpu_dev_calls":len(calls)}
    save(folder/"freeze.json",record)
    print(json.dumps(record),flush=True)


def remote(caps, out):
    import torch
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA device is required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    def get(split,sha):
        return corpus(Path(hf_hub_download(repo_id="sapientinc/maze-30x30-hard-1k",repo_type="dataset",
                                           revision=REV,filename=f"{split}.csv")),sha)
    # All cap/identity checks and CPU/GPU comparisons precede any TEST access.
    train = get("train",TRAIN_SHA)[:DEV_COUNT]
    models = {}
    with torch.inference_mode():
        for cap in caps:
            folder = HERE/f"c{cap}"
            spec,w,check = load(folder)
            fixed = read(folder/"freeze.json")
            if check["model_sha256"] != fixed["model_sha256"] or check["parameters"]>cap:
                raise ValueError("Frozen weights or cap check failed")
            frame = Frame(spec,w,929,cuda=True,template=train[0][0])
            dev,_ = frame.predict([q for q,_ in train])
            if answer_hash(dev) != fixed["cpu_answers_sha256"]:
                raise ValueError(f"NumPy/CUDA DEV mismatch for cap {cap}")
            models[cap] = (frame,fixed)
            print(f"Verified cap {cap} on GPU",flush=True)
        data = get("test",TEST_SHA)
        for cap,(frame,fixed) in models.items():
            solved = 0
            max_steps = 0
            frame.certified_one_hot_steps = 0
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(0,len(data),48):
                q, labels = zip(*data[i:i+48])
                predicted,steps = frame.predict(q)
                solved += sum(p.strip("\n")==a for p,a in zip(predicted,labels))
                max_steps = max(max_steps,steps)
                if cap == 1000:
                    print(f"Scored {i+len(q)} of {len(data)} held-out inputs",flush=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            report = {**fixed,"split":"test","test_sha256":TEST_SHA,"dataset_revision":REV,
                      "solved":solved,"count":len(data),"exact_accuracy":solved/len(data),
                      "test_inference_seconds":seconds,"max_actual_blocks":max_steps,
                      "rounds":929*frame.spec["rounds_per_token"],"head_types":frame.head_types,
                      "device":torch.cuda.get_device_name(),"dtype":"float64","torch":str(torch.__version__),
                      "certified_dynamic_one_hot_head_calls":frame.certified_one_hot_steps,
                      "cpu_gpu_dev_parity":True}
            save(out/f"cap-{cap}.json",report)
            print(f"Cap {cap}: {check_summary(report)}",flush=True)


def check_summary(r):
    return f"{r['parameters']} parameters, {r['solved']}/{r['count']} exact in {r['test_inference_seconds']:.2f}s"


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","gpu"])
    parser.add_argument("caps",nargs="+",type=int,choices=CAPS)
    args=parser.parse_args()
    if args.mode=="prepare":
        for cap in args.caps:
            prepare(cap)
    else:
        remote(args.caps,Path.cwd())
