"""
Controls for the third Google Deep Review (pooled 4-hourly, PMSE unless noted, seeds 0-2).

    .venv/bin/python -W ignore experiments/paper_dr3_checks.py

1. Input representation (configs/paper/final/rangein_*.yaml): the current bar's log range
   in the real part of the price quaternion (groupsets PR, PRMSE) against the pure price
   quaternion (P, PMSE) of final_pool_4h / final_single_4h (ridge).
2. Pre-fusion projection (configs/paper/final/grouproj_pool_4h.yaml): H-LSTM with a learned
   H -> H projection per group (hlp_*) against the H-LSTM (hl_*) and the H-QLSTM (hq_*).

Ratios are means over coins of per-coin MSE ratios of seed-averaged forecasts; DM is the
cross-coin test of evaluate.dm_cross (negative: first model better), Holm-adjusted over the
rows of each table.
Outputs: docs/paper/manuscript/tables/dr3_rangein.{csv,tex}, dr3_grouproj.{csv,tex}
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import LABEL, OUT_T, POOL, run  # noqa: E402
from src.paper.evaluate import dm_cross, holm, seed_mean  # noqa: E402

SEEDS = [0, 1, 2]
RANGEIN = ["lstm", "qlstm", "transformer", "qtransformer", "hl_meta", "hq_meta", "hl_gattn", "hq_gattn"]
LABEL = {**LABEL, "hlp_meta": "H-LSTM+proj. (meta)", "hlp_gattn": "H-LSTM+proj. (group attn.)",
         "hlp_metan": "H-LSTM+proj. (meta, tanh)", "hl_metan": "H-LSTM (meta, tanh)",
         "hq_metan": "H-QLSTM (meta, tanh)"}


def per_coin_mse(df):
    ens = seed_mean(df)
    ens["se"] = (ens.pred - ens.y) ** 2
    return ens.groupby(["dataset", "groupset", "model"]).se.mean()


def ratio(mse, a, b):
    """Mean over coins of MSE(a) / MSE(b); a, b = (groupset, model)."""
    m = mse.unstack(["groupset", "model"])
    r = m[a] / m[b]
    return float(r.mean()), int((r < 1).sum())


def rangein():
    single = run("final_single_4h")
    har = single[(single.groupset == "P") & (single.model == "har")]
    base = run("final_pool_4h")
    base = base[base.seed.isin(SEEDS) & base.model.isin([m + POOL for m in RANGEIN]) & base.groupset.isin(["P", "PMSE"])]
    new = run("final_rangein_pool_4h")
    rid = pd.concat([single[single.model == "ridge"], run("final_rangein_single_4h")])
    df = pd.concat([har, base, new, rid])
    df = df[df.task == "range"]
    mse = per_coin_mse(df)
    rows = []
    pairs = [(m + POOL, gs) for m in RANGEIN for gs in ("P", "PMSE")] + [("ridge", "P"), ("ridge", "PMSE")]
    for model, gs in pairs:
        gr = gs.replace("P", "PR", 1)
        if (gs, model) not in mse.droplevel("dataset").index or (gr, model) not in mse.droplevel("dataset").index:
            continue
        h0 = ratio(mse, (gs, model), ("P", "har"))
        h1 = ratio(mse, (gr, model), ("P", "har"))
        ch = ratio(mse, (gr, model), (gs, model))
        dm = dm_cross(df, (gr, model), (gs, model))
        rows.append(dict(model=model, inputs=gs, har_base=h0[0], har_range=h1[0], beats_har=h1[1],
                         change=ch[0], better=ch[1], dm=dm["dm_stat"], p=dm["p"]))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.to_numpy())
    # twin ratios with the log range in the real part
    tw = []
    for q, r in [("qlstm", "lstm"), ("qtransformer", "transformer"), ("hq_meta", "hl_meta"), ("hq_gattn", "hl_gattn")]:
        for gs in ("PR", "PRMSE"):
            if (gs, q + POOL) in mse.droplevel("dataset").index and (gs, r + POOL) in mse.droplevel("dataset").index:
                rr = ratio(mse, (gs, q + POOL), (gs, r + POOL))
                dm = dm_cross(df, (gs, q + POOL), (gs, r + POOL))
                tw.append(dict(twin=f"{q}/{r}", inputs=gs, ratio=rr[0], q_better=rr[1], dm=dm["dm_stat"], p=dm["p"]))
    tw = pd.DataFrame(tw)
    if len(tw):
        tw["p_holm"] = holm(tw.p.to_numpy())
    return t, tw


def grouproj():
    base = run("final_pool_4h")
    nl = run("final_fusion_nl_pool_4h")
    new = run("final_grouproj_pool_4h")
    single = run("final_single_4h")
    har = single[(single.groupset == "P") & (single.model == "har")]
    df = pd.concat([har, base[base.seed.isin(SEEDS)], nl[nl.seed.isin(SEEDS)], new])
    df = df[(df.task == "range") & ((df.groupset == "PMSE") | (df.model == "har"))]
    mse = per_coin_mse(df)
    rows = []
    for fus in ("meta", "gattn", "metan"):
        hl, hlp, hq = (("PMSE", f"{p}_{fus}{POOL}") for p in ("hl", "hlp", "hq"))
        if hlp not in mse.droplevel("dataset").index:
            continue
        d1, d2 = dm_cross(df, hlp, hl), dm_cross(df, hq, hlp)
        rows.append(dict(fusion=fus, har_hl=ratio(mse, hl, ("P", "har"))[0], har_hlp=ratio(mse, hlp, ("P", "har"))[0],
                         har_hq=ratio(mse, hq, ("P", "har"))[0],
                         hlp_vs_hl=ratio(mse, hlp, hl)[0], dm1=d1["dm_stat"], p1=d1["p"],
                         hq_vs_hlp=ratio(mse, hq, hlp)[0], dm2=d2["dm_stat"], p2=d2["p"],
                         hq_vs_hl=ratio(mse, hq, hl)[0]))
    t = pd.DataFrame(rows)
    if len(t):
        adj = holm(np.r_[t.p1, t.p2])
        t["p1_holm"], t["p2_holm"] = adj[:len(t)], adj[len(t):]
    return t


def star(p):
    return "$^{*}$" if p < 0.05 else ""


def lab(m):
    return LABEL.get(m.replace(POOL, ""), m)


def main():
    pd.set_option("display.width", 250)
    t, tw = rangein()
    t.to_csv(OUT_T / "dr3_rangein.csv", index=False)
    tw.to_csv(OUT_T / "dr3_rangein_twins.csv", index=False)
    print(t.round(4).to_string(), "\n", tw.round(4).to_string())
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Inputs & \multicolumn{2}{c}{MSE / HAR} & With range & Coins & DM \\",
             r" & & pure & with range & / pure & better & \\", r"\midrule"]
    for _, r in t.iterrows():
        name = lab(r.model)
        if r.model != "ridge":
            name = name[:-1] + ", pooled)" if name.endswith(")") else name + " (pooled)"
        lines.append(f"{name} & {r.inputs} & {r.har_base:.3f} & {r.har_range:.3f} & "
                     f"{r.change:.3f} & {r.better}/15 & {r.dm:.2f}{star(r.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / "dr3_rangein.tex").write_text("\n".join(lines) + "\n")

    g = grouproj()
    g.to_csv(OUT_T / "dr3_grouproj.csv", index=False)
    print(g.round(4).to_string())
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Fusion & \multicolumn{3}{c}{MSE / HAR} & \multicolumn{2}{c}{H-LSTM+proj. / H-LSTM} & \multicolumn{2}{c}{\HQ{} / H-LSTM+proj.} \\",
             r" & H-LSTM & +proj. & \HQ{} & ratio & DM & ratio & DM \\", r"\midrule"]
    name = {"meta": "Meta", "gattn": "Group attention", "metan": "Meta + tanh"}
    for _, r in g.iterrows():
        lines.append(f"{name[r.fusion]} & {r.har_hl:.3f} & {r.har_hlp:.3f} & {r.har_hq:.3f} & {r.hlp_vs_hl:.3f} & "
                     f"{r.dm1:.2f}{star(r.p1_holm)} & {r.hq_vs_hlp:.3f} & {r.dm2:.2f}{star(r.p2_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / "dr3_grouproj.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
