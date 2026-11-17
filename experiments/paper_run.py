"""
Paper experiment runner: walk-forward folds x seeds x models x feature sets.

Every (dataset, task, groupset, model, seed, fold) cell writes one .npz with
test and validation predictions; existing cells are skipped, so a run can be
resumed or split across processes with --shard i/n.

    python experiments/paper_run.py --config configs/paper/main.yaml --out results/paper
    python experiments/paper_run.py --config ... --shard 0/4 --device cuda
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np

try:  # the boosting baseline must load its OpenMP runtime before torch (macOS segfault otherwise)
    import lightgbm  # noqa: F401
except ImportError:
    pass
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper.baselines import predict_baseline, predict_baseline_pooled  # noqa: E402
from src.paper.data import make_dataset, walk_forward_folds  # noqa: E402
from src.paper.models import CALENDAR, HIERARCHICAL, build_matched, n_params, target_budget  # noqa: E402
from src.paper.train import fit_predict, fit_predict_pooled, set_seed  # noqa: E402

BASELINES = {"persistence", "mean", "har", "harx", "hars", "harxs", "lgbm", "lgbms", "ridge", "garch", "gjr", "carr"}
warnings.filterwarnings("ignore", category=RuntimeWarning)


POOL = "-pool"


def pools(cfg):
    """freq -> datasets of that frequency (pooled models train on all of them)."""
    by_freq = {}
    for ds in cfg["datasets"]:
        by_freq.setdefault(ds.split("_")[1], []).append(ds)
    return by_freq


def _fits(model, groups):
    """Hierarchical models need >= 2 groups; 4-component fusions need exactly 4; a +cal model
    needs the calendar group as the last group of the input set (it is split off by position)."""
    n_groups = len(groups) if not isinstance(groups, int) else groups
    if model.endswith(CALENDAR):          # the calendar group bypasses the base model
        if not isinstance(groups, int) and groups[-1] != "calendar":
            return False
        n_groups -= 1
    if model.startswith(HIERARCHICAL) and n_groups < 2:
        return False
    if any(model.startswith(p + f) for p in HIERARCHICAL for f in ("meta", "phm")) and n_groups != 4:
        return False
    return True


def cells(cfg):
    # pooled cells: one model per (freq, ...), predictions written per dataset
    for freq, task, gs, model in product(pools(cfg), cfg["tasks"], cfg["groupsets"], cfg.get("pooled_models", [])):
        if not _fits(model, cfg["groupsets"][gs]):
            continue
        for seed in cfg["seeds"]:
            for fold in range(cfg["folds"]["n"]):
                yield f"pool_{freq}", task, gs, model + POOL, seed, fold
    for ds, task, gs, model in product(cfg["datasets"], cfg["tasks"], cfg["groupsets"], cfg["models"]):
        if model in ("har", "harx", "hars", "harxs", "lgbm", "lgbms", "garch", "gjr", "carr") and task != "range":
            continue
        if not _fits(model, cfg["groupsets"][gs]):
            continue
        seeds = cfg["seeds"][:1] if model in BASELINES else cfg["seeds"]  # baselines are deterministic
        for seed in seeds:
            for fold in range(cfg["folds"]["n"]):
                yield ds, task, gs, model, seed, fold


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "results/paper"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = yaml.safe_load(open(args.config))
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    out = Path(args.out) / f"{cfg['name']}_{cfg_hash}"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump({"config": cfg, "hash": cfg_hash, "git": git_sha()}, f, indent=1)

    k, n = map(int, args.shard.split("/"))
    todo = [c for i, c in enumerate(cells(cfg)) if i % n == k]
    print(f"{out} shard {k}/{n}: {len(todo)} cells, device={args.device}", flush=True)

    datasets, folds = {}, {}
    fcfg = cfg["folds"]
    t_start = time.time()

    def load(ds):
        if ds not in datasets:
            coin, freq = ds.split("_")
            datasets[ds] = make_dataset(coin, freq, root=args.data_root, start=cfg.get("start"))
            folds[ds] = walk_forward_folds(datasets[ds].frame.index, fcfg["n"], fcfg["test_days"],
                                           fcfg["val_days"], end=fcfg.get("end"))
        return datasets[ds], folds[ds]

    def save(path, d, y, res, meta, t0):
        idx = d.frame.index.as_unit("ns")
        np.savez(path,
                 test_time=idx[res["test_rows"]].astype("int64"), test_y=y[res["test_rows"]],
                 test_pred=np.asarray(res["test_pred"], np.float32),
                 val_time=idx[res["val_rows"]].astype("int64"), val_y=y[res["val_rows"]],
                 val_pred=np.asarray(res["val_pred"], np.float32),
                 meta=json.dumps({**meta, "seconds": round(time.time() - t0, 1), "device": args.device}))

    for i, (ds, task, gs, model, seed, fold) in enumerate(todo):
        groups = cfg["groupsets"][gs]
        if ds.startswith("pool_"):
            members = pools(cfg)[ds[5:]]
            paths = [out / "preds" / f"{m}__{task}__{gs}__{model}__s{seed}__f{fold}.npz" for m in members]
            if all(p.exists() for p in paths):
                continue
            loaded = [load(m) for m in members]
            t0 = time.time()
            if model[:-len(POOL)] in BASELINES:          # panel HAR family
                ys = [d.frame[f"y_{task}"].to_numpy(np.float32) for d, _ in loaded]
                results = predict_baseline_pooled(model[:-len(POOL)], [d.frame for d, _ in loaded], task,
                                                  [f[fold] for _, f in loaded], cfg["window"][ds[5:]], ds[5:],
                                                  [d.X(groups) for d, _ in loaded], seed=seed)
                for p, (d, _), y, res in zip(paths, loaded, ys, results):
                    save(p, d, y, res, {"pooled": members}, t0)
                print(f"[{i + 1}/{len(todo)}] {ds} {task} {gs} {model} s{seed} f{fold}", flush=True)
                continue
            name, window = model[:-len(POOL)], cfg["window"][ds[5:]]
            budget = target_budget(name, len(groups), cfg["budget"], cfg.get("match"), seq_len=window)
            set_seed(seed)                               # after the twin lookup: it must not shift the seed
            net, hidden = build_matched(name, len(groups), budget, seq_len=window)
            ys = [d.frame[f"y_{task}"].to_numpy(np.float32) for d, _ in loaded]
            results = fit_predict_pooled(net, [d.X(groups) for d, _ in loaded], ys,
                                         [f[fold] for _, f in loaded], cfg["window"][ds[5:]],
                                         cfg["train"], args.device, seed)
            for p, (d, _), y, res in zip(paths, loaded, ys, results):
                save(p, d, y, res, {"hidden": hidden, "params": n_params(net), "budget": budget,
                                    "best_epoch": res["best_epoch"],
                                    "epochs": res["epochs"], "pooled": members}, t0)
            print(f"[{i + 1}/{len(todo)}] {ds} {task} {gs} {model} s{seed} f{fold} "
                  f"{time.time() - t0:.1f}s (elapsed {(time.time() - t_start) / 60:.1f}m)", flush=True)
            continue

        path = out / "preds" / f"{ds}__{task}__{gs}__{model}__s{seed}__f{fold}.npz"
        if path.exists():
            continue
        d, fs = load(ds)
        fo = fs[fold]
        X = d.X(groups)
        y = d.frame[f"y_{task}"].to_numpy(np.float32)
        window = cfg["window"][d.freq]

        t0 = time.time()
        meta = {}
        if model in BASELINES:
            res = predict_baseline(model, d.frame, task, fo, window, d.freq, X=X)
        else:
            budget = target_budget(model, len(groups), cfg["budget"], cfg.get("match"), seq_len=window)
            set_seed(seed)                               # after the twin lookup: it must not shift the seed
            net, hidden = build_matched(model, len(groups), budget, seq_len=window)
            meta = {"hidden": hidden, "params": n_params(net), "budget": budget}
            res = fit_predict(net, X, y, fo, window, cfg["train"], args.device, seed)
            meta.update(best_epoch=res["best_epoch"], epochs=res["epochs"])
        save(path, d, y, res, meta, t0)
        mse = float(np.mean((res["test_pred"] - y[res["test_rows"]]) ** 2))
        print(f"[{i + 1}/{len(todo)}] {path.stem} mse={mse:.5g} {meta} {time.time() - t0:.1f}s "
              f"(elapsed {(time.time() - t_start) / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
