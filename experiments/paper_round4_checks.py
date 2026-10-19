"""
Round-4 review checks, computed from saved forecasts and raw data only.

    .venv/bin/python -W ignore experiments/paper_round4_checks.py seeds      # seed-level twin tests
    .venv/bin/python -W ignore experiments/paper_round4_checks.py bandwidth  # DM bandwidth / bootstrap robustness
    .venv/bin/python -W ignore experiments/paper_round4_checks.py audit      # LunarCrush look-ahead test by period + placebo
    .venv/bin/python -W ignore experiments/paper_round4_checks.py zero_range # bars with H == L (range floored), round 5
    .venv/bin/python -W ignore experiments/paper_round4_checks.py partitions # feature lists of the random partitions R1-R3
    .venv/bin/python -W ignore experiments/paper_round4_checks.py seed_dispersion  # why ensemble ratios leave the seed range
    .venv/bin/python -W ignore experiments/paper_round4_checks.py audit_full # the audit over each series' full history
    .venv/bin/python -W ignore experiments/paper_round4_checks.py audit_sample  # the audit on the model samples (App. B)
    .venv/bin/python -W ignore experiments/paper_round4_checks.py audit_prewhite  # round 6: the audit on AR-pre-whitened features

Outputs: docs/paper/manuscript/tables/review_seed_tests.csv, review_bandwidth.csv, review_audit_period.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import OUT_T, POOL, TWINS, run, setting  # noqa: E402
from src.paper.evaluate import diebold_mariano, dm_table, losses, seed_mean  # noqa: E402

COINS = ["btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "ltc", "dot", "avax", "trx", "bch", "xlm", "atom"]


def cross_diff(df, a, b, task="range"):
    """Cross-sectional mean (over coins) of the per-coin squared-error differential a - b on
    calendar-aligned dates, from seed-averaged forecasts (as in evaluate.dm_cross)."""
    ens = seed_mean(df[df.task == task])
    diffs = []
    for ds, g in ens.groupby("dataset"):
        ga = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
        gb = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
        t = ga.index.intersection(gb.index)
        if len(t) == 0:
            continue
        la = losses(task, ga.loc[t, "y"].to_numpy(), ga.loc[t, "pred"].to_numpy())["se"]
        lb = losses(task, gb.loc[t, "y"].to_numpy(), gb.loc[t, "pred"].to_numpy())["se"]
        diffs.append(pd.Series(la - lb, index=t, name=ds))
    return pd.concat(diffs, axis=1).mean(axis=1).sort_index().to_numpy()


def nw_dm(d, lag):
    """DM statistic of mean(d) = 0 with a Bartlett Newey-West variance and `lag` lags."""
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def block_bootstrap_p(d, block, reps=2000, seed=0):
    """Two-sided moving-block bootstrap p-value of mean(d) = 0 (resample the centred series)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    dc = d - d.mean()
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    means = dc[idx].mean(axis=1)
    return float(np.mean(np.abs(means) >= abs(d.mean())))


# ----------------------------------------------------------------------------- seeds
def main_seeds():
    """For each quaternion/real twin: per matched seed pair (seed k vs seed k) the coin-mean
    MSE ratio and cross-coin DM test; a seed-as-replicate t-test of log ratios and a sign test."""
    twins = [(gs, q, r) for gs in ("PMSE",) for q, r in TWINS] + \
            [("P", "qlstm", "lstm"), ("P", "qtransformer", "transformer")]
    ens_t = pd.read_csv(OUT_T / "final_twins.csv")
    rows = []
    for f in ("4h", "1h", "1d"):
        single, pooled = run(f"final_single_{f}"), run(f"final_pool_{f}")
        for tag, df in (("single", single), ("pooled", pooled)):
            df = df[df.task == "range"]
            sfx = POOL if tag == "pooled" else ""
            for gs, q, r in twins:
                x = df[(df.groupset == gs) & df.model.isin([q + sfx, r + sfx])]
                if x.model.nunique() < 2:
                    continue
                a, b = (gs, q + sfx), (gs, r + sfx)
                ens_ratio = dm_table(x, {"range": b}).rel_loss.mean()
                d = cross_diff(x, a, b)
                ens_stat, ens_p = diebold_mariano(d, np.zeros(len(d)))
                per = []
                for k in sorted(x.seed.unique()):
                    xs = x[x.seed == k]
                    ratio = dm_table(xs, {"range": b}).rel_loss.mean()
                    dk = cross_diff(xs, a, b)
                    st, p = diebold_mariano(dk, np.zeros(len(dk)))
                    per.append((k, ratio, st, p))
                lr = np.log([p[1] for p in per])
                t_p = float(stats.ttest_1samp(lr, 0).pvalue) if len(lr) > 1 else np.nan
                n_up = int((lr > 0).sum())
                sign_p = float(stats.binomtest(n_up, len(lr), 0.5).pvalue)
                ens_dir = np.sign(np.log(ens_ratio))
                same = sum(np.sign(np.log(p[1])) == ens_dir for p in per)
                q_sig = sum((p[3] < .05) and (p[2] < 0) for p in per)
                r_sig = sum((p[3] < .05) and (p[2] > 0) for p in per)
                holm = np.nan
                if gs == "PMSE":
                    from paper_tables_final import LABEL
                    lab = f"{LABEL[q]} / {LABEL[r]}"
                    m = ens_t[(ens_t.setting == f"{tag} {f}") & (ens_t.pair == lab)]
                    holm = float(m.p_holm.iloc[0]) if len(m) else np.nan
                rows.append(dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{r}", n_seeds=len(per),
                                 ens_ratio=ens_ratio, ens_dm=ens_stat, ens_p=ens_p, ens_p_holm=holm,
                                 seed_ratios=";".join(f"{p[1]:.4f}" for p in per),
                                 seed_same_sign=int(same), seed_q_sig=int(q_sig), seed_r_sig=int(r_sig),
                                 ttest_logratio_p=t_p, sign_test_p=sign_p))
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_seed_tests.csv", index=False)
    print(t.drop(columns="seed_ratios").round(4).to_string(index=False))


# ----------------------------------------------------------------------------- bandwidth
def main_bandwidth():
    """Headline cross-coin DM tests with three Newey-West bandwidths and a block bootstrap."""
    comps = []
    for f in ("4h", "1h"):
        base = setting(f)
        base = base[base.task == "range"]
        cal = run(f"final_calendar_pool_{f}")
        cal = cal[cal.task == "range"]
        best = ("PM", "hl_gattn" + POOL)
        comps += [(f, "best network vs HAR", base, best, ("P", "har")),
                  (f, "best network vs HAR-S", base, best, ("PMSE", "hars")),
                  (f, "best network vs HAR-XS", base, best, ("PMSE", "harxs"))]
        both = pd.concat([cal, base[base.model.isin(["hars", "lgbms" + POOL])]])
        comps += [(f, "calendar best vs HAR-S", both, ("PMC", "hl_gattn" + POOL), ("PMSE", "hars")),
                  (f, "calendar best vs LightGBM-S", both, ("PMC", "hl_gattn" + POOL), ("PM", "lgbms" + POOL))]
        if f == "1h":
            comps += [(f, "H-QT/H-T meta (pooled)", base, ("PMSE", "hqt_meta" + POOL), ("PMSE", "ht_meta" + POOL)),
                      (f, "H-QT/H-T concat (pooled)", base, ("PMSE", "hqt_concat" + POOL), ("PMSE", "ht_concat" + POOL))]
        else:
            b3 = base[base.seed.isin([0, 1, 2]) | ~base.model.str.endswith(POOL)]
            comps += [(f, "group attn vs meta, H-LSTM (pooled, seeds 0-2)", b3,
                       ("PMSE", "hl_gattn" + POOL), ("PMSE", "hl_meta" + POOL))]
    b2 = run("final_budget_2000_pool_4h")
    comps.append(("4h", "2k H-QLSTM vs H-LSTM (H=7)", b2[b2.task == "range"],
                  ("PMSE", "hq_meta" + POOL), ("PMSE", "hl_meta" + POOL)))
    rows = []
    for f, name, df, a, b in comps:
        d = cross_diff(df, a, b)
        n = len(d)
        lags = {"nw_n13": int(np.floor(n ** (1 / 3))), "nw_andrews": int(np.floor(4 * (n / 100) ** (2 / 9))),
                "nw_4xn13": 4 * int(np.floor(n ** (1 / 3)))}
        row = dict(freq=f, comparison=name, a=f"{a[1]} {a[0]}", b=f"{b[1]} {b[0]}", n=n,
                   mean_diff=float(d.mean()))
        for k, L in lags.items():
            st, p = nw_dm(d, L)
            row[f"{k}_lags"], row[f"{k}_stat"], row[f"{k}_p"] = L, st, p
        blk = int(round(4 * n ** (1 / 3)))
        row["boot_block"], row["boot_p"] = blk, block_bootstrap_p(d, blk)
        rows.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_bandwidth.csv", index=False)
    print(t.round(5).to_string(index=False))


# ----------------------------------------------------------------------------- audit
def _har_resid(log_range, lags):
    """Residual of an in-sample OLS HAR regression: element t is the innovation of bar t+1
    (log_range_{t+1} minus its fit from information up to bar t)."""
    lr = pd.Series(log_range)
    X = pd.concat([lr.rolling(k).mean() for k in lags], axis=1)
    y = lr.shift(-1)
    ok = X.notna().all(axis=1) & y.notna()
    A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
    beta = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)[0]
    res = pd.Series(np.nan, index=lr.index)
    res[ok] = y[ok].to_numpy() - A @ beta
    return res.to_numpy()


def main_audit(periods=None, out_csv="review_audit_period.csv"):
    """Same-bar vs next-bar correlation of the LunarCrush social/sentiment features with
    returns and HAR volatility innovations, split into 2020-2022 and 2023-2026, plus a placebo
    that shifts one feature forward by one bar (an artificial leak the test should flag).
    Features are the model's transformed inputs (build_features: 30-bar log surprises, /100
    scalings) on each series' full raw history; the HAR innovation is the residual of one
    in-sample OLS HAR fit per coin and frequency over that full history (not per period/fold)."""
    from src.paper.baselines import HAR_LAGS
    from src.paper.data import FFILL_LIMIT, GROUPS, build_features, load_raw
    feats = GROUPS["social"] + GROUPS["sentiment"]
    periods = periods or {"2020-2022": ("2020-01-01", "2022-12-31"), "2023-2026": ("2023-01-01", "2026-10-14")}
    rows = []
    for f in ("1d", "4h", "1h"):
        for coin in COINS:
            try:
                binance, lunar = load_raw(coin, f)
            except FileNotFoundError:
                continue
            fr = build_features(binance, lunar, FFILL_LIMIT[f])
            fr = fr.replace([np.inf, -np.inf], np.nan)
            innov = _har_resid(fr["log_range"].to_numpy(float), HAR_LAGS[f])
            fr["innov_next"] = innov                                   # innovation of bar t+1
            fr["innov_same"] = np.concatenate([[np.nan], innov[:-1]])  # innovation of bar t
            fr["ret_next"] = fr["y_return"]
            fr["placebo"] = fr["galaxy_score"].shift(-1)              # value of bar t+1 at row t: a leak
            for per, (s, e) in periods.items():
                g = fr[(fr.index >= pd.Timestamp(s, tz=fr.index.tz)) & (fr.index <= pd.Timestamp(e, tz=fr.index.tz) + pd.Timedelta("1D"))]
                for col in feats + ["placebo"]:
                    x = g[[col, "ret", "ret_next", "innov_same", "innov_next"]].dropna()
                    if len(x) < 200 or x[col].std() == 0:
                        continue
                    c = x.corr()[col]
                    rows.append(dict(freq=f, coin=coin, period=per, feature=col, n=len(x),
                                     ret_same=c["ret"], ret_next=c["ret_next"],
                                     innov_same=c["innov_same"], innov_next=c["innov_next"]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / out_csv, index=False)
    real = t[t.feature != "placebo"]
    for (f, per), g in real.groupby(["freq", "period"]):
        a = g.assign(ars=g.ret_same.abs(), arn=g.ret_next.abs(), ais=g.innov_same.abs(), ain=g.innov_next.abs())
        ser = a.groupby("coin").agg(ret_same=("ars", "max"), ret_next=("arn", "max"),
                                     innov_same=("ais", "max"), innov_next=("ain", "max"))
        rev = int((ser.innov_next > ser.innov_same).sum())
        feat_rev = int((a.ain > a.ais).sum())
        print(f"{f} {per}: coins={len(ser)}  max|ret_same| mean {ser.ret_same.mean():.3f}  "
              f"max|ret_next| mean {ser.ret_next.mean():.3f} (max {ser.ret_next.max():.3f})  "
              f"max|innov_same| mean {ser.innov_same.mean():.3f}  max|innov_next| mean {ser.innov_next.mean():.3f}  "
              f"series reversed {rev}/{len(ser)}  feature-level reversed {feat_rev}/{len(a)}")
        gs = a[a.feature == "galaxy_score"]
        print(f"   galaxy_score: |ret_same| {gs.ars.mean():.3f} |ret_next| {gs.arn.mean():.3f} "
              f"|innov_same| {gs.ais.mean():.3f} |innov_next| {gs.ain.mean():.3f} reversed {int((gs.ain > gs.ais).sum())}/{len(gs)}")
    pl = t[t.feature == "placebo"]
    for (f, per), g in pl.groupby(["freq", "period"]):
        print(f"PLACEBO {f} {per}: |innov_same| {g.innov_same.abs().mean():.3f} |innov_next| {g.innov_next.abs().mean():.3f} "
              f"|ret_next| {g.ret_next.abs().mean():.3f}  flagged (next>same) {int((g.innov_next.abs() > g.innov_same.abs()).sum())}/{len(g)}")


# ----------------------------------------------------------------------------- round 5: data facts
START = {"1h": "2023-01-01"}  # hourly sample start of the final configs (configs/paper/final/*_1h.yaml)


def main_zero_range():
    """Bars whose target range is zero (H == L, so log(H/L) is floored at 1e-6 and the target is
    log(1e-6) = -13.8), per frequency over the 15 coins, in the training sample of the last
    (largest) fold, its validation block and the four test blocks of the final configs."""
    from src.paper.data import make_dataset, walk_forward_folds
    floor = np.log(1e-6)
    rows = []
    for f in ("4h", "1h", "1d"):
        for coin in COINS:
            d = make_dataset(coin, f, start=START.get(f))
            fo = walk_forward_folds(d.frame.index, 4, 135, 180)
            y = d.frame["y_range"].to_numpy()
            zero = np.isclose(y, floor)
            test = np.concatenate([k["test"] for k in fo])
            for split, idx in (("train (fold 4)", fo[-1]["train"]), ("val (fold 4)", fo[-1]["val"]),
                               ("test (4 blocks)", test), ("all rows", np.arange(len(y)))):
                rows.append(dict(freq=f, coin=coin, split=split, bars=len(idx), zero=int(zero[idx].sum())))
    t = pd.DataFrame(rows)
    agg = t.groupby(["freq", "split"]).agg(bars=("bars", "sum"), zero=("zero", "sum"),
                                           coins_with_zero=("zero", lambda v: int((v > 0).sum())))
    agg["share_pct"] = 100 * agg.zero / agg.bars
    print(agg.to_string())
    print(t[t.zero > 0].to_string(index=False))


def main_partitions():
    """Exact feature assignment of the random 4x4 partitions R1-R3 (src/paper/data.py)."""
    from src.paper.data import GROUPS
    sem = {c: g for g in ("price", "market", "social", "sentiment") for c in GROUPS[g]}
    for s in (1, 2, 3):
        for k in range(4):
            cols = GROUPS[f"rand{s}_{k}"]
            print(f"R{s} group {k + 1}: " + ", ".join(f"{c} ({sem[c][:4]})" for c in cols))


def main_seed_dispersion():
    """For the twin rows whose ensemble ratio lies outside the single-seed range (review_seed_spread.csv):
    MSE(seed mean) = mean seed MSE - mean across-seed forecast variance, so the model whose seeds
    disagree more gains more from averaging. Prints both terms for each model of the pair."""
    sp = pd.read_csv(OUT_T / "review_seed_spread.csv")
    out = sp[(sp.ensemble < sp.seed_min) | (sp.ensemble > sp.seed_max)]
    for r in out.itertuples():
        df = run(f"final_{'pool' if r.setting == 'pooled' else 'single'}_{r.freq}")
        sfx = POOL if r.setting == "pooled" else ""
        for m in (x + sfx for x in r.pair.split("/")):
            g = df[(df.task == "range") & (df.groupset == "PMSE") & (df.model == m)]
            piv = g.pivot_table(index=["dataset", "time"], columns="seed", values="pred")
            y = g.groupby(["dataset", "time"]).y.first().reindex(piv.index)
            seed_mse = (piv.sub(y, axis=0) ** 2).mean().mean()
            ens_mse = ((piv.mean(axis=1) - y) ** 2).mean()
            print(f"{r.freq} {r.setting} {m}: mean seed MSE {seed_mse:.5f}, ensemble MSE {ens_mse:.5f}, "
                  f"across-seed variance {piv.var(axis=1, ddof=0).mean():.5f}, gain {100 * (1 - ens_mse / seed_mse):.2f}%")


def main_audit_full():
    """Round 5: the audit of main_audit over each series' whole history (2020-01-01 to 2026-10-14)."""
    main_audit({"full": ("2020-01-01", "2026-10-14")}, "r5_audit_full.csv")


def main_audit_sample():
    """Round 5: the pooled-period audit on each coin's model sample (make_dataset: common clean
    rows after the last long LunarCrush outage, no hourly 2023 trim), with one in-sample OLS HAR
    per coin and frequency on that sample: mean over coins of the largest |corr| over the eight
    social/sentiment features (and the max over coins of the next-bar return correlation)."""
    from src.paper.baselines import HAR_LAGS
    from src.paper.data import GROUPS, make_dataset
    feats = GROUPS["social"] + GROUPS["sentiment"]
    out = []
    for f in ("1d", "4h", "1h"):
        rows = []
        for c in COINS:
            fr = make_dataset(c, f).frame.copy()
            innov = _har_resid(fr["log_range"].to_numpy(float), HAR_LAGS[f])
            fr["innov_next"], fr["innov_same"] = innov, np.r_[np.nan, innov[:-1]]
            fr["placebo"] = fr["galaxy_score"].shift(-1)              # one-bar-forward leak
            K = fr[feats + ["placebo", "ret", "y_return", "innov_same", "innov_next"]].corr()
            C = K.loc[feats].abs().max()
            rows.append(C)
            out.append(dict(freq=f, coin=c, n=len(fr), **{k: C[k] for k in ("ret", "y_return", "innov_same", "innov_next")},
                            placebo_ret=abs(K.loc["placebo", "ret"]), placebo_next=abs(K.loc["placebo", "y_return"])))
        t = pd.DataFrame(rows)
        print(f"{f}: ret_same {t.ret.mean():.3f}  ret_next {t.y_return.mean():.3f} (max {t.y_return.max():.3f})  "
              f"innov_same {t.innov_same.mean():.3f}  innov_next {t.innov_next.mean():.3f}  "
              f"series with innov_next > innov_same {int((t.innov_next > t.innov_same).sum())}/{len(t)}")
    pd.DataFrame(out).to_csv(OUT_T / "r7_audit_sample.csv", index=False)


def _ar_resid(x, p):
    """Residual of an in-sample OLS AR(p) fit of x on its own lags (pre-whitening): element t is
    the part of x_t not predictable from x_{t-1}, ..., x_{t-p}."""
    s = pd.Series(x)
    X = pd.concat([s.shift(k) for k in range(1, p + 1)], axis=1)
    ok = X.notna().all(axis=1) & s.notna()
    A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
    beta = np.linalg.lstsq(A, s[ok].to_numpy(), rcond=None)[0]
    res = pd.Series(np.nan, index=s.index)
    res[ok] = s[ok].to_numpy() - A @ beta
    return res.to_numpy()


AR_ORDER = {"1d": 7, "4h": 6, "1h": 24}   # one week of daily bars, one day of intraday bars


def main_audit_prewhite():
    """Round 6: the look-ahead test of main_audit on pre-whitened features. Every social/sentiment
    feature (and the one-bar-forward placebo) is replaced by its residual from an in-sample AR(p)
    fit on its own lags (p = AR_ORDER), so persistence no longer lets a shifted series correlate
    with the same bar. A feature whose bar-t innovation correlates more with the bar-(t+1) return
    or HAR innovation than with the bar-t one is flagged."""
    from src.paper.baselines import HAR_LAGS
    from src.paper.data import FFILL_LIMIT, GROUPS, build_features, load_raw
    feats = GROUPS["social"] + GROUPS["sentiment"]
    periods = {"2020-2022": ("2020-01-01", "2022-12-31"), "2023-2026": ("2023-01-01", "2026-10-14")}
    rows = []
    for f in ("1d", "4h", "1h"):
        for coin in COINS:
            try:
                binance, lunar = load_raw(coin, f)
            except FileNotFoundError:
                continue
            fr = build_features(binance, lunar, FFILL_LIMIT[f]).replace([np.inf, -np.inf], np.nan)
            innov = _har_resid(fr["log_range"].to_numpy(float), HAR_LAGS[f])
            fr["innov_next"], fr["innov_same"] = innov, np.r_[np.nan, innov[:-1]]
            fr["placebo"] = fr["galaxy_score"].shift(-1)
            for per, (s, e) in periods.items():
                g = fr[(fr.index >= pd.Timestamp(s, tz=fr.index.tz)) &
                       (fr.index <= pd.Timestamp(e, tz=fr.index.tz) + pd.Timedelta("1D"))].copy()
                for col in feats + ["placebo"]:
                    if g[col].notna().sum() < 200 or g[col].std() == 0:
                        continue
                    g["u"] = _ar_resid(g[col].to_numpy(float), AR_ORDER[f])
                    x = g[["u", "ret", "y_return", "innov_same", "innov_next"]].dropna()
                    if len(x) < 200:
                        continue
                    c = x.corr()["u"]
                    rows.append(dict(freq=f, coin=coin, period=per, feature=col, n=len(x),
                                     ret_same=c["ret"], ret_next=c["y_return"],
                                     innov_same=c["innov_same"], innov_next=c["innov_next"]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r6_audit_prewhite.csv", index=False)
    for (f, per, real), g in t.groupby(["freq", "period", t.feature != "placebo"]):
        a = g.assign(**{k: g[k].abs() for k in ("ret_same", "ret_next", "innov_same", "innov_next")})
        fr_ = int((a.ret_next > a.ret_same).sum())
        fi = int((a.innov_next > a.innov_same).sum())
        tag = "features" if real else "PLACEBO "
        print(f"{tag} {f} {per}: n={len(a)}  |ret_same| {a.ret_same.mean():.3f} |ret_next| {a.ret_next.mean():.3f} "
              f"(max {a.ret_next.max():.3f})  |innov_same| {a.innov_same.mean():.3f} |innov_next| {a.innov_next.mean():.3f} "
              f"(max {a.innov_next.max():.3f})  flagged ret {fr_}/{len(a)}  innov {fi}/{len(a)}")
        if real:
            gs = a[a.feature == "galaxy_score"]
            print(f"   galaxy_score: |ret_same| {gs.ret_same.mean():.3f} |ret_next| {gs.ret_next.mean():.3f} "
                  f"|innov_same| {gs.innov_same.mean():.3f} |innov_next| {gs.innov_next.mean():.3f} "
                  f"flagged ret {int((gs.ret_next > gs.ret_same).sum())}/{len(gs)} "
                  f"innov {int((gs.innov_next > gs.innov_same).sum())}/{len(gs)}")


if __name__ == "__main__":
    {"audit_prewhite": main_audit_prewhite, "audit_sample": main_audit_sample, "audit_full": main_audit_full, "seed_dispersion": main_seed_dispersion, "seeds": main_seeds, "bandwidth": main_bandwidth, "audit": main_audit, "zero_range": main_zero_range,
     "partitions": main_partitions}[sys.argv[1]]()
