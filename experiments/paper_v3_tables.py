"""
Summary of the grid for candidate A (configs/paper/v3/a_*.yaml), computed from saved forecasts.

    .venv/bin/python -W ignore experiments/paper_v3_tables.py [grid|bars|seeds|small ...]
                                              tables/v3_grid.csv, v3_grid_bars.csv, v3_grid_seeds.csv, v3_small.csv

Candidate A is hq_gattn on price with range, market and calendar (PRMC); hl_gattn is its real twin. Every
loss ratio is the MSE over all coins and test bars of one model divided by that of the other (as in the
paper). Rows of v3_grid.csv (pooled 4-hour bars, three seeds): MSE relative to HAR of the quaternion and of
the real model of a control, the twin ratio of the control, and the control against candidate A.
A run that is not on disk yet is skipped with a note.
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v2_round2 as R  # noqa: E402
import paper_v2_summary as S  # noqa: E402

OUT = "docs/paper/manuscript_v2/tables"
A, AR = ("PRMC", "hq_gattn-pool"), ("PRMC", "hl_gattn-pool")
A_RUN = "results/v3local/v3_gattn_range_pool_4h_*"
HAR_RUN = "results/final_fa/out/final_single_4h_*"
V3 = "results/v3*/out/v3_a_{}_*"

# (block, label, run pattern, quaternion key or None, real key or None)
GRID = [
    ("Inputs", "without the range in the price quaternion", "results/final_fc/out/final_calendar_pool_4h_*", ("PMC", "hq_gattn-pool"), ("PMC", "hl_gattn-pool")),
    ("Inputs", "without the calendar group", V3.format("inputs_pool_4h"), ("PRM", "hq_gattn-pool"), ("PRM", "hl_gattn-pool")),
    ("Inputs", "without the market group", V3.format("inputs_pool_4h"), ("PRC", "hq_gattn-pool"), ("PRC", "hl_gattn-pool")),
    ("Inputs", "with social and sentiment groups", V3.format("inputs_pool_4h"), ("PRMSEC", "hq_gattn-pool"), ("PRMSEC", "hl_gattn-pool")),
    ("Calendar", "additive term instead of a group", V3.format("addcal_pool_4h"), ("PRMC", "hq_gattn+cal-pool"), ("PRMC", "hl_gattn+cal-pool")),
    ("Fusion", "concatenation", V3.format("fusion_pool_4h"), ("PRMC", "hq_concat-pool"), ("PRMC", "hl_concat-pool")),
    ("Fusion", "dense", V3.format("fusion_pool_4h"), ("PRMC", "hq_dense-pool"), ("PRMC", "hl_dense-pool")),
    ("Depth", "two layers", V3.format("depth_pool_4h"), ("PRMC", "hq_gattn-L2-pool"), ("PRMC", "hl_gattn-L2-pool")),
    ("Encoder", "flat QLSTM / LSTM", V3.format("flat_pool_4h"), ("PRMC", "qlstm-pool"), ("PRMC", "lstm-pool")),
    ("Budget", "2,000 parameters", V3.format("budget_2000_pool_4h"), A, AR),
    ("Budget", "5,000 parameters", V3.format("budget_5000_pool_4h"), A, AR),
    ("Budget", "80,000 parameters", V3.format("budget_80000_pool_4h"), A, AR),
    ("Learning rate", "3e-3", V3.format("lr3e3_pool_4h"), A, AR),
    ("Learning rate", "3e-4", V3.format("lr3e4_pool_4h"), A, AR),
    ("Training", "one model per coin", V3.format("single_4h"), ("PRMC", "hq_gattn"), ("PRMC", "hl_gattn")),
    ("Benchmarks on the same inputs", "DLinear", V3.format("baselines_pool_4h"), None, ("PRMC", "dlinear-pool")),
    ("Benchmarks on the same inputs", "patch Transformer (PatchTST)", V3.format("baselines_pool_4h"), None, ("PRMC", "patchtst-pool")),
    ("Benchmarks on the same inputs", "inverted Transformer (iTransformer)", V3.format("baselines_pool_4h"), None, ("PRMC", "itransformer-pool")),
    ("Benchmarks on the same inputs", "Transformer", V3.format("baselines_pool_4h"), None, ("PRMC", "transformer-pool")),
    ("Benchmarks on the same inputs", "QCNN / CNN, 6 convolutions", V3.format("baselines_pool_4h"), ("PRMC", "qcnn-L6-pool"), ("PRMC", "cnn-L6-pool")),
]


def _run(pattern: str, keys: list[tuple], tag: str = "", cells_needed: int = S.N_CELLS) -> pd.DataFrame | None:
    dirs = glob.glob(pattern)
    if len(dirs) != 1:
        return None
    x = S.run(dirs[0])
    x = x[[(g, m) in keys for g, m in zip(x.groupset, x.model)]]
    cells = x.groupby(["groupset", "model"]).apply(lambda g: len(g[["dataset", "seed", "fold"]].drop_duplicates()), include_groups=False)
    if len(cells) < len(set(keys)) or (cells < cells_needed).any():
        return None                                         # the run is not complete
    return R._tag(x, tag)


def grid() -> pd.DataFrame:
    base = pd.concat([_run(A_RUN, [A, AR]), _run(HAR_RUN, [R.HAR], cells_needed=60)], ignore_index=True)   # HAR has one seed
    rows = []
    st0 = R.Store([base])
    rows.append(dict(block="Candidate A", control="hq_gattn / hl_gattn on PRMC", quaternion=st0.vs_har(A), real=st0.vs_har(AR),
                     **{f"twin_{k}": v for k, v in st0.pair(A, AR).items() if k in ("ratio", "better", "p", "ci_lo", "ci_hi", "equivalent")}))
    for block, label, pattern, q, r in GRID:
        keys = [k for k in (q, r) if k is not None]
        x = _run(pattern, keys, "@c")
        if x is None:
            print("not available:", block, "/", label)
            continue
        tag = lambda k: (k[0], k[1] + "@c")
        st = R.Store([base, x])
        row = dict(block=block, control=label)
        if q is not None:
            v = st.pair(tag(q), A)
            row.update(quaternion=st.vs_har(tag(q)), vs_A=v["ratio"], vs_A_better=v["better"], vs_A_p=v["p"], vs_A_lo=v["ci_lo"], vs_A_hi=v["ci_hi"])
        if r is not None:
            v = st.pair(tag(r), A)
            row.update(real=st.vs_har(tag(r)), real_vs_A=v["ratio"], real_vs_A_better=v["better"], real_vs_A_p=v["p"])
        if q is not None and r is not None:
            row.update({f"twin_{k}": v for k, v in st.pair(tag(q), tag(r)).items() if k in ("ratio", "better", "p", "ci_lo", "ci_hi", "equivalent")})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "v3_grid.csv"), index=False)
    return t


SEED_RUNS = [A_RUN, V3.format("seeds_pool_4h")]


def seeds() -> pd.DataFrame:
    """Candidate A and its twin with all trained seeds, as paper_v2_round2.seeds does for the pairs of the paper."""
    if not glob.glob(SEED_RUNS[1]):
        print("not available: seeds 3-9")
        return pd.DataFrame()
    pairs = [("Candidate A / real twin", A, AR, SEED_RUNS),
             ("Candidate A, all seeds / LightGBM-S, lagged", A, ("PM", "lgbmsl-pool"), SEED_RUNS + ["results/r8_local/final_lgbmlags_4h_r8"])]
    t = _seed_pairs(pairs)
    t.to_csv(os.path.join(OUT, "v3_grid_seeds.csv"), index=False)
    return t


SMALL = (500, 1000, 2000)


def small() -> pd.DataFrame:
    """Small-model test of the algebra (configs/paper/v3/a_small_*_pool_4h.yaml, plan fixed there before the run):
    candidate A against its real twin at 500, 1,000 and 2,000 parameters with ten seeds, and each model at a
    small budget against itself at 20,000 parameters. The interval of the rule is the joint seed-and-time
    interval of the ratio for averages of five networks (k5_joint_lo, k5_joint_hi; five is the largest average
    that can be drawn without replacement from ten seeds in both arms). `advantage` is True when it lies below 1."""
    pairs = []
    for b in SMALL:
        pattern = V3.format(f"small_{b}_pool_4h")
        if len(glob.glob(pattern)) != 1:
            print("not available:", b)
            continue
        tag = f"@{b}"
        qa, ra = (A[0], A[1] + tag), (AR[0], AR[1] + tag)
        pairs += [(f"{b:,} parameters: H-QLSTM / real twin", qa, ra, [(pattern, tag)]),
                  (f"H-QLSTM, {b:,} / 20,000 parameters", qa, A, [(pattern, tag)] + SEED_RUNS),
                  (f"real twin, {b:,} / 20,000 parameters", ra, AR, [(pattern, tag)] + SEED_RUNS)]
    if not pairs:
        return pd.DataFrame()
    t = _seed_pairs(pairs)
    t["advantage"] = t.k5_joint_hi < 1
    t.to_csv(os.path.join(OUT, "v3_small.csv"), index=False)
    twins = t[t.pair.str.contains("real twin$")]
    print(f"rule: the quaternion model has an advantage at {int(twins.advantage.sum())} of {len(twins)} budgets (two needed)")
    return t[["pair", "seeds_a", "seeds_b", "mse_har_a", "mse_har_b", "ratio_all", "better", "p", "ci_lo", "ci_hi", "k5_joint_lo", "k5_joint_hi", "advantage"]]


def _seed_pairs(pairs: list) -> pd.DataFrame:
    """paper_v2_round2.seeds for the given pairs (it writes its table into a scratch folder, read back here)."""
    import pathlib
    old, keep = R.SEED_PAIRS, R.OUT
    R.SEED_PAIRS, R.OUT = pairs, pathlib.Path(OUT) / "_v3_tmp"
    try:
        R.OUT.mkdir(exist_ok=True)
        R.seeds()
        return pd.read_csv(R.OUT / "v2_seeds10.csv")
    finally:
        for f in glob.glob(os.path.join(OUT, "_v3_tmp", "*")):
            os.remove(f)
        os.rmdir(os.path.join(OUT, "_v3_tmp"))
        R.SEED_PAIRS, R.OUT = old, keep


BAR_RUNS = {   # bar length -> (run of candidate A, runs with HAR and the benchmarks of that bar length)
    "1h": (V3.format("pool_1h"), ["results/final_fb/out/final_single_1h_*", "results/r8_local/final_harsx_1h_r8", "results/r8_local/final_lgbmlags_1h_r8",
                                  "results/v2f/out/v2_cal_pool_1h_*"]),
    "1d": (V3.format("pool_1d"), ["results/final_fa/out/final_single_1d_*", "results/final_local/final_har_season_1d_*", "results/r8_local/final_lgbmlags_1d_r8",
                                  "results/v2f/out/v2_cal_pool_1d_*"]),
}
BAR_ROWS = [("Candidate A", A), ("Candidate A, real twin", AR), ("LightGBM-S, lagged", ("PM", "lgbmsl-pool")), ("HAR-S x", ("P", "hars_x")),
            ("HAR-S", ("PMSE", "hars")), ("Proposed model of the paper", R.PROPOSED)]


def bars() -> pd.DataFrame:
    rows = []
    for freq, (own, others) in BAR_RUNS.items():
        if len(glob.glob(own)) != 1:
            print("not available:", freq)
            continue
        keep = {k for _, k in BAR_ROWS} | {R.HAR}
        parts = []
        for pattern in [own] + others:
            x = S.run(pattern)
            parts.append(x[[(g, m) in keep for g, m in zip(x.groupset, x.model)]])
        st = R.Store([pd.concat(parts, ignore_index=True)])
        for label, key in BAR_ROWS:
            if not st.has(key):
                continue
            r = dict(bars=freq, model=label, mse_har=st.vs_har(key))
            if key != A:
                v = st.pair(A, key)
                r.update(A_over_model=v["ratio"], A_better=v["better"], p=v["p"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"])
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "v3_grid_bars.csv"), index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40)
    fns = {f.__name__: f for f in (grid, bars, seeds, small)}
    for fn in ([fns[a] for a in sys.argv[1:]] or fns.values()):
        print(f"\n=== {fn.__name__}")
        t = fn()
        if len(t):
            print(t.round(4).to_string(index=False))
