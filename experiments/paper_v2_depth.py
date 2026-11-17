"""
V2 depth and convolution grid (configs/paper/v2/depth_*.yaml): tables of the models and twin pairs.

    .venv/bin/python -W ignore experiments/paper_v2_depth.py [--results results/v2/out] [--partial]

Outputs (docs/paper/manuscript_v2/):
  tables/v2_depth_models.csv   one row per (budget, model): family, algebra, depth, width, parameters,
                               MSE relative to HAR (mean over coins of the seed-ensemble ratio), coins
                               better than HAR, median stopping epoch
  tables/v2_depth_twins.csv    one row per quaternion/real pair: mean MSE ratio over coins, its range over
                               single seeds, coins where the quaternion model is better, Holm-significant
                               coins in each direction, cross-coin Diebold-Mariano test, Holm over the table
The figure of MSE/HAR against depth is drawn from v2_depth_models.csv by experiments/paper_v2_figures.py
(top row of fig_scope.pdf).

Every statistic follows paper v1 (experiments/paper_tables_final.py: twin_table, _vs_har), so the depth-1
recurrent rows are directly comparable with tables/final_twins.csv and final_budget.csv.

A model enters the tables only when all its cells (coins x seeds x folds) are present; --partial lifts
this for a look at a run in progress (such numbers are not results).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.paper.evaluate import dm_cross, dm_table, holm, load_run  # noqa: E402
from src.paper.models import split_depth  # noqa: E402

OUT_T = Path("docs/paper/manuscript_v2/tables")
POOL = "-pool"
GS = "PMSE"
RUNS = {20_000: "v2_depth_pool_4h", 80_000: "v2_depth_80000_pool_4h"}

# base name -> (family, quaternion?, default depth)
FAMILY = {
    "lstm": ("LSTM", False, 1), "qlstm": ("LSTM", True, 1),
    "hl_meta": ("H-LSTM (meta)", False, 1), "hq_meta": ("H-LSTM (meta)", True, 1),
    "cnn": ("CNN", False, 4), "qcnn": ("CNN", True, 4),
    "hc_meta": ("H-CNN (meta)", False, 4), "hqc_meta": ("H-CNN (meta)", True, 4),
}
FAMILIES = ["LSTM", "H-LSTM (meta)", "CNN", "H-CNN (meta)"]


def describe(model: str) -> dict:
    """'hq_meta-L2-pool' -> family, algebra and depth."""
    base, depth = split_depth(model[:-len(POOL)] if model.endswith(POOL) else model)
    family, quaternion, default = FAMILY[base]
    return dict(family=family, algebra="quaternion" if quaternion else "real", depth=depth or default)


def run_dir(results: str, name: str) -> str | None:
    dirs = [d for d in glob.glob(os.path.join(results, f"{name}_*")) if glob.glob(os.path.join(d, "preds", "*.npz"))]
    if len(dirs) > 1:
        raise RuntimeError(f"several runs for {name}: {dirs}")
    return dirs[0] if dirs else None


def cell_meta(directory: str) -> pd.DataFrame:
    """Width, parameter count and stopping epoch of every network cell (from the npz meta)."""
    rows = []
    for path in glob.glob(os.path.join(directory, "preds", f"*{POOL}__s*__f*.npz")):
        ds, _, _, model, seed, fold = os.path.basename(path)[:-4].split("__")
        meta = json.loads(str(np.load(path)["meta"]))
        if "params" in meta:
            rows.append(dict(dataset=ds, model=model, seed=int(seed[1:]), fold=int(fold[1:]),
                             hidden=meta["hidden"], params=meta["params"], best_epoch=meta["best_epoch"]))
    return pd.DataFrame(rows)


def complete_models(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Pooled models with every (coin, seed, fold) cell present."""
    want = len(cfg["datasets"]) * len(cfg["seeds"]) * cfg["folds"]["n"]
    have = df[df.model.str.endswith(POOL)].groupby("model").apply(
        lambda g: len(g[["dataset", "seed", "fold"]].drop_duplicates()), include_groups=False)
    return sorted(have[have == want].index)


def load(results: str, partial: bool):
    """{budget: (forecasts incl. HAR, cell meta)} for the runs that exist."""
    out, har = {}, None
    for budget, name in RUNS.items():
        d = run_dir(results, name)
        if d is None:
            continue
        df = load_run(d)
        cfg = json.load(open(os.path.join(d, "config.json")))["config"]
        if har is None and (df.model == "har").any():
            har = df[df.model == "har"]
            n_har = len(har[["dataset", "fold"]].drop_duplicates())
            if n_har < len(cfg["datasets"]) * cfg["folds"]["n"] and not partial:
                raise SystemExit(f"[{name}] HAR benchmark incomplete ({n_har} cells): MSE/HAR would cover "
                                 "only part of the test period; wait for the run or pass --partial")
        keep = sorted(m for m in df.model.unique() if m.endswith(POOL)) if partial else complete_models(df, cfg)
        skipped = sorted(set(m for m in df.model.unique() if m.endswith(POOL)) - set(keep))
        if skipped:
            print(f"[{name}] incomplete, left out: {', '.join(skipped)}")
        out[budget] = (df[df.model.isin(keep)], cell_meta(d))
    if har is None:
        raise SystemExit("no HAR cells found: run configs/paper/v2/depth_pool_4h.yaml first")
    return {b: (pd.concat([har, df]), meta) for b, (df, meta) in out.items()}


def models_table(data) -> pd.DataFrame:
    rows = []
    for budget, (df, meta) in data.items():
        d = dm_table(df, {"range": (GS, "har")})
        for model, g in d.groupby("model"):
            m = meta[meta.model == model]
            rows.append(dict(budget=budget, model=model, **describe(model),
                             hidden=int(m.hidden.iloc[0]), params=int(m.params.iloc[0]),
                             mse_har=g.rel_loss.mean(), better=int((g.rel_loss < 1).sum()), n=len(g),
                             best_epoch=float(m.best_epoch.median())))
    t = pd.DataFrame(rows).sort_values(["budget", "family", "depth", "algebra"]).reset_index(drop=True)
    t.to_csv(OUT_T / "v2_depth_models.csv", index=False)
    return t


def seed_ratio_range(df: pd.DataFrame, q: str, r: str) -> tuple[float, float]:
    """Smallest and largest coin-mean MSE ratio q/r over the single seeds (seed s against seed s)."""
    x = df[df.model.isin([q, r])].assign(se=lambda v: (v.pred - v.y) ** 2)
    mse = x.groupby(["model", "seed", "dataset"]).se.mean().unstack("model")
    ratio = (mse[q] / mse[r]).groupby("seed").mean()
    return float(ratio.min()), float(ratio.max())


def twins_table(data, models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, (df, _) in data.items():
        have = models[models.budget == budget]
        for (family, depth), g in have.groupby(["family", "depth"]):
            if set(g.algebra) != {"quaternion", "real"}:
                continue
            q = g[g.algebra == "quaternion"].iloc[0]
            r = g[g.algebra == "real"].iloc[0]
            x = df[(df.groupset == GS) & df.model.isin([q.model, r.model])]
            t = dm_table(x, {"range": (GS, r.model)})
            t["p_coins"] = holm(t.p)
            c = dm_cross(x, (GS, q.model), (GS, r.model))
            lo, hi = seed_ratio_range(x, q.model, r.model)
            rows.append(dict(budget=budget, family=family, depth=depth, q_model=q.model, r_model=r.model,
                             q_params=q.params, r_params=r.params, q_har=q.mse_har, r_har=r.mse_har,
                             ratio=t.rel_loss.mean(), seed_min=lo, seed_max=hi,
                             q_better=int((t.rel_loss < 1).sum()), n=len(t),
                             q_sig_better=int(((t.rel_loss < 1) & (t.p_coins < .05)).sum()),
                             r_sig_better=int(((t.rel_loss > 1) & (t.p_coins < .05)).sum()),
                             dm=c["dm_stat"], p=c["p"]))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t.p)              # family: every twin row of this table
        t["family"] = pd.Categorical(t.family, FAMILIES)
        t = t.sort_values(["budget", "family", "depth"]).reset_index(drop=True)
    t.to_csv(OUT_T / "v2_depth_twins.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v2/out")
    ap.add_argument("--partial", action="store_true", help="include models whose cells are not all present")
    args = ap.parse_args()
    data = load(args.results, args.partial)
    models = models_table(data)
    twins = twins_table(data, models)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(models.round(4).to_string(index=False))
    print()
    print(twins.drop(columns=["q_model", "r_model"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
