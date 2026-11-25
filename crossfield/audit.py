"""Regrade the published cross-field GPU outputs directly from raw records with NumPy."""
import argparse
from collections import Counter
import gzip
from itertools import combinations
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_evidence import authenticate_run, digest, read, require

INPUTS = ("experiment.py", "protocol.json", "checks.json", "inflection.py", "diffraction.py", "auction.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("crossfield-results.json", "crossfield-predictions.npz")
SIZES = {"inflection": [2, 4, 8, 16], "diffraction": [4, 8, 16, 32], "auction": [2, 4, 6, 8]}
CLASSES = {"inflection": 27, "diffraction": 257, "auction": 22}
ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
CASES = ("nominative", "locative", "ablative", "accusative", "genitive")
SUFFIX = ("", "lar", "ler", "da", "de", "ta", "te", "dan", "den", "tan", "ten",
          "ı", "i", "u", "ü", "yı", "yi", "yu", "yü", "ın", "in", "un", "ün", "nın", "nin", "nun", "nün")


def inflection(data):
    words = data["requests"]
    require(set(data) == {"requests"} and 1 <= len(words) <= 32, "Invalid request batch")
    tokens = np.zeros((len(words), 26), dtype="<i8")
    labels, stems, forms = [], [], []
    # Four-way harmony columns run back-unrounded, front-unrounded,
    # back-rounded, front-rounded. Odd columns are front vowels.
    vowel_column = {v: i for i, letters in enumerate(("aı", "ei", "ou", "öü")) for v in letters}
    for i, req in enumerate(words):
        require(set(req) == {"stem", "number", "case"}, "Invalid inflection request")
        stem, number, case = req["stem"], req["number"], req["case"]
        require(type(stem) is str and 1 <= len(stem) <= 24 and all(c in ALPHABET for c in stem)
                and any(c in vowel_column for c in stem), "Invalid unchanged stem")
        require(number in ("singular", "plural") and case in CASES, "Unsupported number or case")
        plural = int(number == "plural")
        tokens[i, :len(stem)] = [1+ALPHABET.index(c) for c in stem]
        tokens[i, -2:] = plural, CASES.index(case)
        col = [vowel_column[c] for c in stem if c in vowel_column][-1]
        front = col % 2
        number_code = 1 + front if plural else 0
        vowel_end = int(not plural and stem[-1] in vowel_column)
        hard = int(not plural and stem[-1] in "fstkçşhp")
        if plural:
            col = front   # -lar / -ler supplies an unrounded vowel.
        if case == "nominative":
            case_code = 0
        elif case in ("locative", "ablative"):
            case_code = (3 if case == "locative" else 7) + 2*hard + front
        else:
            case_code = (11 if case == "accusative" else 19) + 4*vowel_end + col
        labels.extend((number_code, case_code))
        stems.append(stem)
        forms.append(stem + SUFFIX[number_code] + SUFFIX[case_code])
    return tokens, np.asarray(labels), {"stems": stems, "forms": forms}


def diffraction(data):
    sites, refl = data["sites"], data["reflections"]
    require(set(data) == {"sites", "reflections"} and 1 <= len(sites) <= 8 and 1 <= len(refl) <= 32,
            "Invalid crystal record dimensions")
    require(all(len(s) == 4 and all(type(v) is int for v in s) and all(0 <= v <= 3 for v in s[:3])
                and s[3] in (1, 2) for s in sites) and len({tuple(s[:3]) for s in sites}) == len(sites),
            "Invalid crystal atom sites")
    require(all(len(h) == 3 and all(type(v) is int and abs(v) <= 4 for v in h) for h in refl),
            "Invalid reflection")
    tokens = np.zeros((8+len(refl), 4), dtype="<i8")
    tokens[:len(sites)] = sites
    tokens[8:, :3] = refl
    # Pairwise expansion of F F*, evaluated with exact quarter-turn cosines.
    baseline = sum(s[3]**2 for s in sites)
    cross_terms = [(tuple(x-y for x,y in zip(a[:3], b[:3])), 2*a[3]*b[3]) for a,b in combinations(sites, 2)]
    cosine = (1, 0, -1, 0)
    answer = [baseline + sum(weight*cosine[sum(p*d for p,d in zip(h, delta)) % 4]
                             for delta, weight in cross_terms) for h in refl]
    positions = {tuple(h): i for i,h in enumerate(refl)}
    opposite_pairs = []
    for h,i in positions.items():
        opposite = tuple(-v for v in h)
        if h < opposite and opposite in positions:
            j = positions[opposite]
            require(answer[i] == answer[j], "Opposite reflections violate the real-amplitude symmetry")
            opposite_pairs.append((i, j))
    return tokens, np.asarray(answer), {"pairs": opposite_pairs}


def auction(data):
    bids = data["bids"]
    require(set(data) == {"bids"} and 1 <= len(bids) <= 8, "Invalid bid record")
    offers, amounts = [], []
    for offer in bids:
        bundle, value = offer["bundle"], offer["value"]
        require(set(offer) == {"bundle", "value"} and type(bundle) is str and bundle
                and bundle == "".join(c for c in "ABCD" if c in bundle)
                and type(value) is int and 1 <= value <= 20, "Invalid bundle or bid")
        offers.append(set(bundle))
        amounts.append(value)
    tokens = np.array([[int(item in bundle) for item in "ABCD"] + [v] for bundle,v in zip(offers, amounts)], dtype="<i8")
    feasible = []
    for count in range(len(bids) + 1):
        for selected in combinations(range(len(bids)), count):
            demanded = [item for i in selected for item in offers[i]]
            if len(set(demanded)) == len(demanded):
                feasible.append((sum(amounts[i] for i in selected), sum(1 << i for i in selected), selected))
    welfare = max(entry[0] for entry in feasible)
    best = [entry for entry in feasible if entry[0] == welfare]
    winners = min(best, key=lambda entry: entry[1])[2]
    codes = np.zeros(len(bids), dtype=np.int64)
    for i in winners:
        alternative = max(value for value, _, indices in feasible if i not in indices)
        charge = alternative - sum(amounts[j] for j in winners if j != i)
        require(0 <= charge <= amounts[i], "Violation of pivot payment bounds")
        codes[i] = 1 + charge
    greedy, used = 0, set()
    for i in sorted(range(len(bids)), key=lambda i: (-amounts[i], i)):
        if used.isdisjoint(offers[i]):
            used.update(offers[i])
            greedy += amounts[i]
    return tokens, codes, {"greedy_suboptimal": greedy < welfare, "tied_optima": len(best) > 1}


ORACLES = {"inflection": inflection, "diffraction": diffraction, "auction": auction}


def pack(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def tensor_ledgers(lock):
    expected_shapes = {
        "inflection": {"orthography": (len(ALPHABET)+1, 5), "address": (2,), "classifier": (27,10), "bias": (27,)},
        "diffraction": {"address": (2,), "roots": (4,2), "pool_gain": (), "knots": (16,), "slopes": (16,)},
        "auction": {"query": (2,), "readout": (4,)},
    }
    require(set(lock["models"]) == set(expected_shapes), "Frozen model set differs")
    result = {}
    with np.load(HERE / "weights.npz", allow_pickle=False) as archive:
        expected = {f"{t}/{k}" for t, spec in expected_shapes.items() for k in spec}
        require(set(archive.files) == expected and len(archive.files) == len(expected), "Weight names differ")
        for t,spec in expected_shapes.items():
            model = lock["models"][t]
            require(set(model["tensors"]) == set(spec), f"Unexpected tensors for {t}")
            count = nonzero = 0
            for name,shape in spec.items():
                value = archive[f"{t}/{name}"]
                nnz = int(np.count_nonzero(value))
                info = {"shape": list(shape), "dtype": "float32", "parameter": True, "nonzero": nnz,
                        "sha256": digest(value.tobytes())}
                require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all()
                        and model["tensors"][name] == info, f"Invalid frozen tensor {t}/{name}")
                count += value.size
                nonzero += nnz
            require(count == model["parameters"], f"Invalid parameter count for {t}")
            result[t] = {"parameters": count, "nonzero": nonzero, "tensors": len(spec)}
    return result


def frozen():
    lock = read(HERE / "freeze.json")
    protocol = read(HERE / "protocol.json")
    for key, filename in (("weights","weights.npz"), ("cases","cases.json.gz"), ("protocol","protocol.json"),
                          ("checks","checks.json")):
        require(lock[key + "_sha256"] == digest((HERE / filename).read_bytes()), f"Freeze mismatch for {filename}")
    require(protocol["study"] == "crossfield" and protocol["seed"] == 2026112504 and protocol["batch"] == 8
            and protocol["repeats"] == 3 and protocol["per_size"] == {"dev": 8,"test": 64}
            and set(protocol["tasks"]) == set(ORACLES), "Invalid protocol")
    for task, sizes in SIZES.items():
        cfg = protocol["tasks"][task]
        require(cfg["dev"] == sizes[:2] and cfg["test"] == sizes and cfg["classes"] == CLASSES[task],
                f"Invalid protocol for {task}")
    ledgers = tensor_ledgers(lock)
    checks = read(HERE / "checks.json")
    require(Counter(c["task"] for c in checks) == {"inflection": 1, "diffraction": 4, "auction": 3}
            and len({c["name"] for c in checks}) == 8, "Invalid named fixtures")
    seen = {task: set() for task in SIZES}
    def fingerprint(tokens):
        return tokens.shape, digest(tokens.tobytes(order="C"))
    for task in ORACLES:
        tokens, _, _ = ORACLES[task](read(HERE / "examples" / f"{task}.json"))
        seen[task].add(fingerprint(tokens))
    named = {}
    for case in checks:
        task = case["task"]
        tokens, labels, props = ORACLES[task](case["input"])
        seen[task].add(fingerprint(tokens))
        if task == "inflection":
            require(len(props["forms"]) == 12 and props["forms"] == case["forms"], "Worked surface forms differ")
        else:
            require(labels.tolist() == case["labels"], f"Worked labels differ for {case['name']}")
        named[case["name"]] = (task, labels, props)
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    expected = {f"{split}/{task}/{size}" for task in SIZES for split in ("dev","test")
                for size in protocol["tasks"][task][split]}
    require(set(cases) == set(lock["cohorts"]) == expected, "Missing or unexpected cohort")
    cohorts = {}
    for key, rows in cases.items():
        split, task, size = key.split("/")
        require(len(rows) == protocol["per_size"][split] and digest(pack(rows)) == lock["cohorts"][key],
                f"Invalid size/hash in {key}")
        gold, properties = [], []
        for row in rows:
            require(set(row) == {"input","gold"}, "Unexpected oracle metadata")
            tokens, labels, props = ORACLES[task](row["input"])
            fp = fingerprint(tokens)
            require(fp not in seen[task], f"Repeated input in {key}")
            seen[task].add(fp)
            require(len(tokens) == int(size) + (8 if task == "diffraction" else 0), f"Wrong input length in {key}")
            target = np.asarray(row["gold"])
            require(target.dtype.kind in "iu" and np.array_equal(target, labels), f"Oracle mismatch in {key}")
            gold.append(labels)
            properties.append(props)
        cohorts[key] = np.stack(gold), properties
    return lock, ledgers, cohorts, named


def validate_pred(pred, gold, task, shapes, name):
    require(pred.shape == gold.shape and list(pred.shape) == shapes[name] and pred.dtype == np.int16
            and pred.min() >= -1 and pred.max() < CLASSES[task], f"Invalid prediction array for {name}")


def evaluate(run):
    results = read(run / "output" / OUTPUTS[0])
    identity, shapes = authenticate_run(run, results, folder=HERE, inputs=INPUTS, outputs=OUTPUTS,
                                        stage_name="crossfield")
    lock, ledgers, cohorts, named = frozen()
    require(all(lock[key] == results[key] for key in ("cases_sha256", "weights_sha256")), "Unmatched runtime inputs")
    rows = {(r["task"],r["size"]): r for r in results["conditions"]}
    pairs = [(t,n) for t, sizes in SIZES.items() for n in sizes]
    require(len(results["conditions"]) == len(pairs) and set(rows) == set(pairs), "GPU condition coverage differs")
    checked, checked_fixtures = [], {}
    cats = {task: Counter() for task in SIZES}
    stems, forms = set(), set()
    with np.load(run / "output" / OUTPUTS[1], allow_pickle=False) as archive:
        keys = {f"{t}_{n}" for t,n in pairs} | {f"check_{name}" for name in named}
        require(set(archive.files) == set(shapes) == keys and len(archive.files) == len(keys), "Prediction names differ")
        require(set(results["checks"]) == set(named), "GPU named fixture coverage differs")
        for name, (task, gold, props) in named.items():
            key = f"check_{name}"
            pred = archive[key]
            validate_pred(pred, gold, task, shapes, key)
            ok = pred == gold
            entry = {"task": task, "correct": bool(ok.all())}
            if task == "inflection":
                surfaces = [s + SUFFIX[number] + SUFFIX[case] if min(number,case) >= 0 else None
                            for s,(number,case) in zip(props["stems"], pred.reshape(-1, 2), strict=True)]
                entry["forms"] = surfaces
                entry["correct"] = surfaces == props["forms"]
            require(entry == results["checks"][name], f"Worked example score differs for {name}")
            checked_fixtures[name] = {**entry, "labels": gold.size, "correct_labels": int(ok.sum())}
        for task, size in pairs:
            gold, props = cohorts[f"test/{task}/{size}"]
            key = f"{task}_{size}"
            pred = archive[key]
            validate_pred(pred, gold, task, shapes, key)
            ok = pred == gold
            exact = ok.all(axis=1)
            row = rows[task,size]
            scores = {"parameters": ledgers[task]["parameters"], "records": len(pred), "exact_records": int(exact.sum()),
                      "labels": gold.size, "correct_labels": int(ok.sum()), "nonfinite": int((pred < 0).sum())}
            require(all(value == row[k] for k,value in scores.items()), f"Scores disagree for {key}")
            times = row["seconds"]
            require(len(times) == 3 and all(np.isfinite(t) and t > 0 for t in times)
                    and row["median_seconds"] == median(times), f"Timing mismatch for {key}")
            checked.append({"task": task, "size": size, **scores, "seconds": times, "median_seconds": median(times)})
            c = cats[task]
            if task == "inflection":
                c["requests"] += gold.size//2
                c["forms_correct"] += int(ok.reshape(-1,2).all(axis=1).sum())
                for entry in props:
                    stems.update(entry["stems"])
                    forms.update(entry["forms"])
            elif task == "diffraction":
                c["reflections"] += gold.size
                for name, flags in (("zero", gold == 0), ("nonzero", gold != 0)):
                    c[name] += int(flags.sum())
                    c[name+"_correct"] += int((flags & ok).sum())
                for i, entry in enumerate(props):
                    c["friedel_pairs"] += len(entry["pairs"])
                    c["friedel_pairs_correct"] += sum(int(ok[i,a] and ok[i,b]) for a,b in entry["pairs"])
            else:
                for name in ("greedy_suboptimal", "tied_optima"):
                    group = np.asarray([entry[name] for entry in props])
                    c[name] += int(group.sum())
                    c[name+"_correct"] += int((group & exact).sum())
                zero_prices = gold == 1
                c["zero_price_winners"] += int(zero_prices.sum())
                c["zero_price_winners_correct"] += int((zero_prices & ok).sum())
    cats["inflection"].update(distinct_stems=len(stems), distinct_surface_forms=len(forms))
    totals = {task: {key: sum(row[key] for row in checked if row["task"] == task)
                     for key in ("records", "exact_records", "labels", "correct_labels", "nonfinite")}
              for task in SIZES}
    return {"verified": True, "run": identity, "model_ledgers": ledgers, "frozen_cohorts_checked": len(cohorts),
            "development_records_checked": sum(len(a) for k,(a,_) in cohorts.items() if k.startswith("dev/")),
            "test_records_checked": sum(r["records"] for r in checked), "labels_regraded": sum(r["labels"] for r in checked),
            "exact_test_records": sum(r["exact_records"] for r in checked),
            "correct_test_labels": sum(r["correct_labels"] for r in checked), "task_totals": totals,
            "checks": checked_fixtures, "test_categories": {task: dict(c) for task,c in cats.items()}, "conditions": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    evidence = evaluate(args.version_folder.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes((json.dumps(evidence, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode())
    print(f"Regraded {evidence['test_records_checked']} test records, {evidence['labels_regraded']:,} labels "
          f"and {len(evidence['checks'])} separate worked examples.")
