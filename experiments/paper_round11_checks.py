"""Round-11 checks (internal deep review round 11): the strongest linear benchmark under the primary loss.

  harxsx_save   HAR-XS x: HAR regressors, the current bar's PMSE features and the (time of day x day of
                week) interaction dummies of the target bar (day-of-week dummies on daily bars), single-coin
                OLS per training block -> results/r11_local/final_harxsx_<freq>_r11 (model harxs_x)
  frontier      every forecast clipped to its coin's training-fold target range (the ex-ante filter of
                Supplementary Section S7); mean per-coin MSE ratio to HAR of HAR-S, HAR-S x, HAR-XS,
                HAR-XS x, the lagged pooled LightGBM-S and the best calendar network, and the calendar
                network against each linear benchmark (mean per-coin ratio, coins better, cross-coin DM)
                -> tables/r11_frontier.csv
  val_select    validation selection of the best pooled network on all validation dates and on the
                validation dates before the first test block only (each fold's validation block contains
                the previous fold's test block) -> tables/r11_val_select.csv
  mcs_final     90% MCS of paper_round10_checks.mcs_tuned with HAR-XS x (unclipped, like every other member)
                added -> tables/r11_mcs_final.csv

Run:  .venv/bin/python experiments/paper_round11_checks.py harxsx_save frontier val_select mcs_final
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_round7_checks import CAL_NET, _frames, _interactions, _load  # noqa: E402
from paper_round8_checks import LGBM_LAG  # noqa: E402
from paper_tables_final import OUT_T  # noqa: E402

FREQS = ("4h", "1h", "1d")
HARXSX_DIR = "results/r11_local/final_harxsx_{f}_r11"
HARXSX = ("PMSE", "harxs_x")
HAR, HARS, HARSX, HARXS = ("P", "har"), ("PMSE", "hars"), ("P", "hars_x"), ("PMSE", "harxs")


def harxsx_save(freqs=FREQS):
    from src.paper.baselines import HAR_LAGS, _har_design, _ols
    for f in freqs:
        out = os.path.join(HARXSX_DIR.format(f=f), "preds")
        os.makedirs(out, exist_ok=True)
        json.dump({"config": {"name": f"final_harxsx_{f}", "note": "paper_round11_checks.harxsx_save"}},
                  open(os.path.join(os.path.dirname(out), "config.json"), "w"), indent=1)
        fr, win = _frames(f)
        for ds, (d, folds) in fr.items():
            y = d.frame["y_range"].to_numpy(float)
            D = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[f]),
                                 d.X(["price", "market", "social", "sentiment"]).astype(float),
                                 _interactions(d.frame.index, d.step)])
            idx = d.frame.index if d.frame.index.tz is None else d.frame.index.tz_convert(None)
            t = idx.as_unit("ns").asi8
            for k, fo in enumerate(folds):
                ok = fo["train"][~np.isnan(D[fo["train"]]).any(1)]
                b = _ols(D[ok], y[ok])
                te, va = (r[r >= win - 1] for r in (fo["test"], fo["val"]))
                np.savez(os.path.join(out, f"{ds}__range__{HARXSX[0]}__{HARXSX[1]}__s0__f{k}.npz"),
                         test_time=t[te], test_y=y[te].astype("float32"),
                         test_pred=(b[0] + D[te] @ b[1:]).astype("float32"), val_time=t[va],
                         val_y=y[va].astype("float32"), val_pred=(b[0] + D[va] @ b[1:]).astype("float32"),
                         meta=np.array(json.dumps(dict(freq=f, fold=k, model="harxs_x"))))
        print(f, "saved", flush=True)


def _bounds(f):
    """(dataset, fold) -> training-fold min / max of the target (folds of the single-coin config)."""
    fr, _ = _frames(f)
    out = {}
    for ds, (d, folds) in fr.items():
        y = d.frame["y_range"].to_numpy(float)
        for k, fo in enumerate(folds):
            out[(ds, k)] = (y[fo["train"]].min(), y[fo["train"]].max())
    return out


def frontier(freqs=FREQS):
    from src.paper.evaluate import dm_cross
    rows = []
    for f in freqs:
        keys = {HAR, HARS, HARSX, HARXS, HARXSX, LGBM_LAG, CAL_NET}
        names = [f"final_single_{f}", f"final_har_season_{f}", f"final_harx_{f}", f"final_harsx_{f}",
                 f"final_harxsx_{f}", f"final_lgbmlags_{f}", f"final_calendar_pool_{f}"]
        df = _load(names, lambda g, m: (g, m) in keys)
        df = df[~((df.groupset == CAL_NET[0]) & (df.model == CAL_NET[1]) & (df.seed > 2))]
        df = df[~((df.model == LGBM_LAG[1]) & (df.seed > 2))]
        e = df.groupby(["dataset", "groupset", "model", "fold", "time"], as_index=False).agg(
            y=("y", "first"), pred=("pred", "mean"))
        B = _bounds(f)
        lo = np.array([B[(d, k)][0] for d, k in zip(e.dataset, e.fold)])
        hi = np.array([B[(d, k)][1] for d, k in zip(e.dataset, e.fold)])
        n_clip = (e.pred.to_numpy() != np.clip(e.pred, lo, hi)).groupby([e.groupset, e.model]).sum()
        e["pred"] = np.clip(e.pred, lo, hi)
        present = [k for k in keys if ((e.groupset == k[0]) & (e.model == k[1])).any()]

        def ratios(a, b):
            per = []
            for ds, g in e.groupby("dataset"):
                A = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
                Bb = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
                t = A.index.intersection(Bb.index)
                per.append(np.mean((A.loc[t, "pred"] - A.loc[t, "y"]) ** 2)
                           / np.mean((Bb.loc[t, "pred"] - Bb.loc[t, "y"]) ** 2))
            return np.array(per)

        for k in present:
            r = ratios(k, HAR)
            rows.append(dict(freq=f, a=f"{k[1]} {k[0]}", b="har P", mean_ratio=r.mean(), better=int((r < 1).sum()),
                             n=len(r), p=np.nan, n_clipped=int(n_clip.get(k, 0))))
        for b in (HARS, HARSX, HARXS, HARXSX, LGBM_LAG):
            if b not in present or CAL_NET not in present:
                continue
            r = ratios(CAL_NET, b)
            c = dm_cross(e.assign(task="range", seed=0)[[(g, m) in (CAL_NET, b) for g, m in zip(e.groupset, e.model)]],
                         CAL_NET, b)
            rows.append(dict(freq=f, a=f"{CAL_NET[1]} {CAL_NET[0]}", b=f"{b[1]} {b[0]}", mean_ratio=r.mean(),
                             better=int((r < 1).sum()), n=len(r), p=c["p"], n_clipped=np.nan))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r11_frontier.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


def val_select():
    rows = []
    runs = [(f, f"final_calendar_pool_{f}") for f in FREQS] + [("4h", "final_pool_4h")]
    for f, name in runs:
        df = _load([name], lambda g, m: m.endswith("-pool"), splits=("val", "test"))
        t0 = df[df.split == "test"].time.min()
        e = df.groupby(["split", "dataset", "groupset", "model", "time"], as_index=False).agg(
            y=("y", "first"), pred=("pred", "mean"))
        e["se"] = (e.pred - e.y) ** 2
        for label, x in (("val_all", e[e.split == "val"]), ("val_pretest", e[(e.split == "val") & (e.time < t0)]),
                         ("test", e[e.split == "test"])):
            mse = x.groupby(["groupset", "model"]).se.mean().sort_values()
            for rank, ((g, m), v) in enumerate(mse.items(), 1):
                rows.append(dict(freq=f, run=name, selection=label, rank=rank, groupset=g, model=m, mse=v))
        print(f, name, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r11_val_select.csv", index=False)
    print(t[t["rank"] <= 3].to_string(index=False))
    return t


def mcs_final(freqs=FREQS):
    """paper_round10_checks.mcs_tuned (seeds 0-2, same exclusions) with the HAR-XS x forecasts added."""
    from arch.bootstrap import MCS
    from paper_review_checks import RUNS, ensemble, load_both
    from paper_round10_checks import _tuned_key
    rows = []
    for f in freqs:
        tname, tuned = _tuned_key(f)
        names = RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}", f"final_lgbmlags_{f}",
                           f"final_harsx_{f}", tname, f"final_harxsx_{f}"]
        E = ensemble(load_both(names), seeds=[0, 1, 2])
        E = E[(E.split == "test") & ~E.model.isin(["mean"])]
        E["se"] = (E.pred - E.y) ** 2
        L = E.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        mcs = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
        mcs.compute()
        cols = list(L.columns)
        names_in = [f"{cols[i][1]} {cols[i][0]}" for i in mcs.included]
        pv = mcs.pvalues.squeeze()
        p = lambda k: float(pv.get(cols.index(k), np.nan)) if k in cols else np.nan
        rows.append(dict(freq=f, n_models=L.shape[1], n_mcs=len(names_in), members="; ".join(names_in),
                         has_harxsx=HARXSX in cols, p_harxsx=p(HARXSX), p_lgbm_lag=p(LGBM_LAG),
                         p_cal_net=p(CAL_NET)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT_T / "r11_mcs_final.csv", index=False)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        {"harxsx_save": harxsx_save, "frontier": frontier, "val_select": val_select, "mcs_final": mcs_final}[arg]()
