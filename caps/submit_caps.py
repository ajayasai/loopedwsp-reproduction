"""Submit frozen cap models to a private Kaggle GPU job and archive its receipt."""
import argparse
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from submit import ACCELERATOR, OWNER, cli_executable


def run(payload, hashes, caps):
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    import zlib
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    scratch = Path("/kaggle/temp")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        root = Path(folder)
        for key, blob in payload.items():
            data = zlib.decompress(base64.b64decode(blob))
            if hashlib.sha256(data).hexdigest() != hashes[key]:
                raise ValueError("Packaged file hash mismatch")
            p = root / key
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data)
        output = Path("/kaggle/working")
        proc = subprocess.run([sys.executable,"-u",str(root/"experiment.py"),"gpu",*map(str,caps)],cwd=output)
        if proc.returncode:
            raise RuntimeError(f"Cap evaluation failed with status {proc.returncode}")
        (output/"manifest.json").write_text(json.dumps(hashes,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("caps",type=int,nargs="+",choices=(1000,5000,10000,20000,40000))
    p.add_argument("--push",action="store_true")
    args=p.parse_args()
    caps=sorted(set(args.caps))
    label="-".join(str(cap//1000) for cap in caps)
    slug=f"loopedwsp-maze-caps-{label}"
    folder=ROOT/"kaggle/packages"/slug
    folder.mkdir(parents=True,exist_ok=True)
    files=[HERE/"frame.py",HERE/"experiment.py"]
    for cap in caps:
        files.extend(HERE/f"c{cap}"/name for name in ("weights.npz","network.json","freeze.json"))
    content={f.relative_to(HERE).as_posix():f.read_bytes() for f in files}
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in content.items()}
    payload={k:base64.b64encode(zlib.compress(v)).decode() for k,v in content.items()}
    source=inspect.getsource(run)+f"\nrun({payload!r}, {hashes!r}, {caps!r})\n"
    compile(source,"main.py","exec")
    (folder/"main.py").write_text(source,encoding="utf-8",newline="\n")
    meta={"id":f"{OWNER}/{slug}","title":f"LoopedWSP Maze Caps {label}","code_file":"main.py",
          "language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,
          "enable_internet":True,"enable_tpu":False,"dataset_sources":[],"kernel_sources":[],"competition_sources":[]}
    (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    if not args.push:
        return
    result=subprocess.run([cli_executable(),"kernels","push","-p",str(folder),"--accelerator",ACCELERATOR,
                           "-t","10800"],capture_output=True,text=True,errors="replace",
                          env={**os.environ,"PYTHONUTF8":"1"})
    msg=result.stdout+result.stderr
    print(msg,flush=True)
    match=re.search(r"Kernel version (\d+) successfully pushed",msg)
    if not match or result.returncode:
        raise RuntimeError("Kaggle submission was not acknowledged")
    version=int(match.group(1))
    target=ROOT/"kaggle/runs"/slug/f"v{version}"
    target.mkdir(parents=True,exist_ok=True)
    for name in ("main.py","kernel-metadata.json"):
        shutil.copy2(folder/name,target/name)
    receipt={"id":meta["id"],"version":version,"caps":caps,"hashes":hashes,
             "url":f"https://www.kaggle.com/code/{meta['id']}/versions/{version}"}
    (target/"submission.json").write_text(json.dumps(receipt,indent=2))
    print(target,flush=True)


if __name__=="__main__":
    main()
