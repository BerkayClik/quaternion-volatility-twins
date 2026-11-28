"""
Cascade of gradient boosting and the network (asked for by the advisor, 2026-11-18): the lagged LightGBM-S
forecasts first, and the network is trained on what it leaves, the residual y - LightGBM. The forecast of the
cascade is LightGBM plus the network. Fixed before the run:

  stage 1  pooled LightGBM-S with lagged exchange features, exactly the benchmark of the paper (three seeds,
           averaged). On validation and test rows its forecast is that of the model fitted on the training
           block. On training rows the forecast is out of fold: the training period is cut into five blocks of
           equal length in time, and the rows of a block are forecast by a model fitted on the other four
           (early stopping on the validation block, as always). Without this the residual on the training
           rows would be that of a model that has seen them, and the second stage would learn from a target
           that is smaller than the one it meets on new data.
  stage 2  candidate A (hq_gattn on price with range, market, calendar) and its real twin, trained with the
           protocol of the paper on the residual as target (configs/paper/v3/a_cascade_pool_4h.yaml; the
           runner subtracts the stage-1 forecast from the target and adds it back to its forecasts).

Cascades with HAR as the first stage (asked for by the advisor on the same day: the combination of the quaternion
network with HAR; fixed before the run): stage 1 is HAR (three lags of the log range) or HAR-S x
(HAR with one dummy per cell of the week), OLS per coin as in the paper, out of fold over the same five blocks
on the training rows; stage 2 as above (configs a_cascade_har_pool_4h.yaml, a_cascade_harsx_pool_4h.yaml).
They are read by experiments/paper_v3_hybrid_har.py together with the averages and the stacking with HAR.

First two stages of the cascade HAR -> LightGBM -> network (experiments/paper_v3_triple.py; `harlgbm`): HAR as
above, then the lagged LightGBM-S fitted on the residual y - HAR, out of fold over the same five blocks on the
training rows (where HAR is out of fold too). The saved forecast is the sum of the two stages.

    .venv/bin/python -W ignore experiments/paper_v3_cascade.py offsets [har|harsx|harlgbm] [confirm] [fold ...]
                                                        <data root>/offsets/v3_cascade[_har|_harsx|_har_lgbm]_4h
    .venv/bin/python -W ignore experiments/paper_v3_cascade.py check [har|harsx] [confirm]   stage 1 on test rows = the saved benchmark
    .venv/bin/python -W ignore experiments/paper_v3_cascade.py eval                          tables/v3_cascade.csv
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

FREQ, WINDOW, K_BLOCKS = "4h", 42, 5
NAME = "v3_cascade_4h"
# first stage -> (folder of its forecasts, key of the saved benchmark it must reproduce, runs of that benchmark on folds 1-4)
STAGES = {"lgbm": (NAME, ("PM", "lgbmsl-pool"), ["results/r8_local/final_lgbmlags_4h_r8"]),
          "har": ("v3_cascade_har_4h", ("P", "har"), ["results/final_fa/out/final_single_4h_*"]),
          "harsx": ("v3_cascade_harsx_4h", ("P", "hars_x"), ["results/r8_local/final_harsx_4h_r8"]),
          "harlgbm": ("v3_cascade_har_lgbm_4h", None, None)}        # two stages: no saved benchmark to reproduce


def stage1(frames: dict, fold_index: int, seeds=(0, 1, 2), base: dict | None = None) -> dict:
    """Stage-1 forecast for every row of the training, validation and test blocks of one fold, in the units of
    each coin: {dataset: Series(time -> forecast)}. frames: dataset -> (Dataset, [folds]).
    base (optional): the forecast of an earlier stage, {dataset: Series} as returned by stage1_ols. LightGBM is
    then fitted on the residual y - base, and the result is base plus LightGBM (the first two stages of a cascade)."""
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from src.paper.baselines import HAR_LAGS, _design, _lgbm
    from src.paper.data import time_blocks
    coins = []
    for ds, (d, folds) in frames.items():
        fold = folds[fold_index]
        X = d.X(["price", "market"])
        Xl = np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]])
        D = _design("lgbms", d.frame, FREQ, Xl)
        b = np.zeros(len(D)) if base is None else base[ds].to_numpy(float)     # NaN where the earlier stage has no forecast
        y = d.frame["y_range"].to_numpy(float) - b
        ok = fold["train"][~np.isnan(D[fold["train"]]).any(1) & np.isfinite(b[fold["train"]])]
        mx, sx = D[ok].mean(0), D[ok].std(0)
        sx[sx < 1e-8] = 1.0
        my, sy = y[ok].mean(), y[ok].std()
        sel = lambda r: r[r >= WINDOW - 1]
        coins.append(dict(ds=ds, D=(D - mx) / sx, z=(y - my) / sy, my=my, sy=sy, fit=ok, train=fold["train"], base=b,
                          val=sel(fold["val"]), test=sel(fold["test"]), time=d.frame.index, fold=fold))
    edges = time_blocks([c["time"] for c in coins], [c["fold"] for c in coins], K_BLOCKS)
    Xva = np.vstack([c["D"][c["val"]] for c in coins])
    zva = np.concatenate([c["z"][c["val"]] for c in coins])
    out = {c["ds"]: np.full(len(c["time"]), np.nan) for c in coins}
    total = {c["ds"]: np.zeros(len(c["time"])) for c in coins}
    for seed in seeds:
        for j in range(K_BLOCKS + 1):                       # the five held-out blocks, then the fit on all training rows
            held = lambda c, rows: (c["time"][rows] >= edges[j]) & (c["time"][rows] < edges[j + 1]) if j < K_BLOCKS else np.zeros(len(rows), bool)
            A = np.vstack([c["D"][c["fit"][~held(c, c["fit"])]] for c in coins])
            b = np.concatenate([c["z"][c["fit"][~held(c, c["fit"])]] for c in coins])
            m = _lgbm(A, b, Xva, zva, seed=seed)
            for c in coins:
                rows = c["train"][held(c, c["train"])] if j < K_BLOCKS else np.concatenate([c["val"], c["test"]])
                if len(rows):
                    total[c["ds"]][rows] += c["my"] + c["sy"] * m.predict(c["D"][rows])
        print(f"fold {fold_index} seed {seed} done", flush=True)
    for c in coins:
        rows = np.concatenate([c["train"], c["val"], c["test"]])
        out[c["ds"]][rows] = c["base"][rows] + total[c["ds"]][rows] / len(seeds)
    return {c["ds"]: pd.Series(out[c["ds"]], index=c["time"]) for c in coins}


def stage1_ols(frames: dict, fold_index: int, seasonal: bool) -> dict:
    """As stage1, for HAR (seasonal=False) or HAR-S x (True): OLS per coin, as the benchmarks of the paper are fitted.
    Rows whose regressors are not complete (the first bars of a coin) have no forecast (NaN); no window ends there."""
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _har_design, _ols
    from src.paper.data import time_blocks
    edges = time_blocks([d.frame.index for d, _ in frames.values()], [folds[fold_index] for _, folds in frames.values()], K_BLOCKS)
    out = {}
    for ds, (d, folds) in frames.items():
        fold, time = folds[fold_index], d.frame.index
        D = _har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[FREQ])
        if seasonal:
            D = np.column_stack([D, _interactions(time, d.step)])
        y = d.frame["y_range"].to_numpy(float)
        tr = fold["train"]
        ok = tr[~np.isnan(D[tr]).any(1)]
        pred = np.full(len(time), np.nan)
        for j in range(K_BLOCKS):                             # training rows: forecast by the fit on the other four blocks
            held = (time[ok] >= edges[j]) & (time[ok] < edges[j + 1])
            b = _ols(D[ok[~held]], y[ok[~held]])
            pred[ok[held]] = b[0] + D[ok[held]] @ b[1:]
        b = _ols(D[ok], y[ok])                                # validation and test rows: the fit on the training block
        rows = np.concatenate([r[r >= WINDOW - 1] for r in (fold["val"], fold["test"])])
        pred[rows] = b[0] + D[rows] @ b[1:]
        out[ds] = pd.Series(pred, index=time)
    return out


def offsets(stage: str = "lgbm", confirm: bool = False, only: list[int] | None = None):
    """Writes the stage-1 forecasts, one file per coin and fold. `only`: folds to compute (one process per fold)."""
    if confirm:
        import paper_v3_confirm as C
        frames = {ds: (d, [fold]) for ds, (d, fold) in C.frames().items()}
        root, n_folds = "data_post", 1
    else:
        from paper_round7_checks import _frames
        frames, _ = _frames(FREQ)
        root, n_folds = "data", 4
    out = os.path.join(root, "offsets", STAGES[stage][0])
    os.makedirs(out, exist_ok=True)
    for k in (range(n_folds) if only is None else only):
        if stage == "lgbm":
            forecasts = stage1(frames, k)
        elif stage == "harlgbm":
            forecasts = stage1(frames, k, base=stage1_ols(frames, k, seasonal=False))
        else:
            forecasts = stage1_ols(frames, k, seasonal=stage == "harsx")
        for ds, s in forecasts.items():
            idx = s.index if s.index.tz is None else s.index.tz_convert(None)
            np.savez(os.path.join(out, f"{ds}__f{k}.npz"), time=idx.as_unit("ns").asi8, offset=s.to_numpy("float32"))
    json.dump({"note": "paper_v3_cascade.offsets", "stage": stage, "blocks": K_BLOCKS, "seeds": [0, 1, 2] if "lgbm" in stage else None},
              open(os.path.join(out, "info.json"), "w"))
    print("saved", out)


def check(stage: str = "lgbm", root: str = "data"):
    """On test rows the stage-1 forecast must be the saved forecast of the benchmark (same model)."""
    import paper_v2_round2 as R
    name, key, runs = STAGES[stage]
    if key is None:
        print(f"{stage}: no saved benchmark to compare with")
        return
    ens = R._splits([key], runs if root == "data" else ["results/v3confirm/bench"])
    ens = ens[ens.split == "test"]
    worst = 0.0
    for (ds, fold), g in ens.groupby(["dataset", "fold"]):
        z = np.load(os.path.join(root, "offsets", name, f"{ds}__f{fold}.npz"))
        s = pd.Series(z["offset"], index=pd.to_datetime(z["time"], utc=True))
        worst = max(worst, float((s.reindex(g.time.to_numpy()).to_numpy() - g.pred.to_numpy()).__abs__().max()))
    print(f"{stage}: largest gap between stage 1 and the saved benchmark on test rows: {worst:.2e}")


CASCADE = {"quaternion": ("PRMC", "hq_gattn-pool"), "real": ("PRMC", "hl_gattn-pool")}
AVG = ("HY", "blend_avg")


def _period(runs: list[str], pattern: str) -> pd.DataFrame | None:
    """Test forecasts of one period in the long format of the analysis: the two parts, their average, HAR and the cascades."""
    import paper_v2_summary as S
    import paper_v3_hybrid as H
    dirs = glob.glob(pattern)
    if len(dirs) != 1:
        return None
    w = H.base(runs)
    te = w[w.split == "test"]
    long = lambda pred, key: pd.DataFrame({"dataset": te.dataset, "task": "range", "groupset": key[0], "model": key[1], "seed": 0,
                                           "fold": te.fold, "time": te.time, "y": te.y, "pred": pred})
    parts = [long(te[s], H.KEYS[s]) for s in ("N", "G", "HAR")] + [long((te.N + te.G) / 2, AVG)]
    c = S.run(dirs[0])
    for name, key in CASCADE.items():
        x = c[(c.groupset == key[0]) & (c.model == key[1])]
        assert len(x[["dataset", "seed", "fold"]].drop_duplicates()) == 15 * 3 * te.fold.nunique(), f"{name}: the run is not complete"
        parts.append(x.assign(groupset="CA", model=name))
    return pd.concat(parts, ignore_index=True)


def evaluate() -> pd.DataFrame:
    """The cascade on the four test blocks, on the new fold and on all five, against its two parts and their average."""
    import paper_v2_round2 as R
    import paper_v3_hybrid as H
    frames = {"four test blocks": _period(H.TEST_RUNS, "results/v3*/out/v3_a_cascade_pool_4h_*"),
              "33 days after the sample": _period(H.CONFIRM_RUNS, "results/v3confirm/out/v3_confirm_cascade_pool_4h_*")}
    if all(f is not None for f in frames.values()):
        frames["all five blocks"] = pd.concat([frames["four test blocks"], frames["33 days after the sample"].assign(fold=4)], ignore_index=True)
    rows = []
    for period, df in frames.items():
        if df is None:
            print("not available:", period)
            continue
        st = R.Store([df])
        others = (("A", H.KEYS["N"]), ("G", H.KEYS["G"]), ("avg", AVG))
        for label, key in [("candidate A", H.KEYS["N"]), ("LightGBM-S, lagged", H.KEYS["G"]), ("average of the two", AVG),
                           ("cascade: LightGBM, then H-QLSTM on the residual", ("CA", "quaternion")),
                           ("cascade: LightGBM, then H-LSTM on the residual", ("CA", "real"))]:
            r = dict(period=period, model=label, mse_har=st.vs_har(key))
            for name, other in others:
                if key != other:
                    v = st.pair(key, other)
                    r.update({f"vs_{name}": v["ratio"], f"vs_{name}_lo": v["ci_lo"], f"vs_{name}_hi": v["ci_hi"], f"vs_{name}_better": v["better"], f"p_{name}": v["p"]})
            rows.append(r)
        v = st.pair(("CA", "quaternion"), ("CA", "real"))
        rows.append(dict(period=period, model="cascade, quaternion over real second stage", mse_har=np.nan, vs_A=np.nan,
                         twin=v["ratio"], twin_lo=v["ci_lo"], twin_hi=v["ci_hi"], twin_better=v["better"], p_twin=v["p"]))
    t = pd.DataFrame(rows)
    t.to_csv("docs/paper/manuscript_v2/tables/v3_cascade.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(t.round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    stage = next((a for a in rest if a in STAGES), "lgbm")
    if cmd == "offsets":
        offsets(stage, confirm="confirm" in rest, only=[int(a) for a in rest if a.isdigit()] or None)
    elif cmd == "check":
        check(stage, "data_post" if "confirm" in rest else "data")
    else:
        evaluate()
