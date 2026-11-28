"""
Checks for manuscript v5 after an automated external review (2026-11-28; the review is kept in
docs/paper/review/google_2026-11-28/review.md). Five checks that the review asks for and that no table of the paper
answers. Plan fixed here and committed, with this code and the configuration of check 5, before any of them was
run. Known before: every result of the paper (tables/v3_*, v4_*, v5_*.csv). The rules of the paper are not changed:
the two tests stay Welch over runs and Diebold-Mariano over time, the level constant of QLIKE and the range limits
stay as they are. Each check is a sensitivity analysis next to them, reported in the text in one or two sentences
whatever it shows.

  1 paired   The test over runs treats the five runs of two models as independent samples, although run k of a hybrid
             contains run k of its members and the two versions of the network share their seeds. Beside Welch's
             test: the paired t-test on the five differences (run k against run k), the correlation of the losses of
             the runs, and the verdict of the two-test rule with the paired test in the place of Welch's. MSE, on the
             six data sets of the comparison of the two versions (development data on four-hour, hourly and daily
             bars; the three held-out sets). Comparisons: the hybrid against LightGBM, against its network, against
             the real twin alone and against the hybrid with the twin; the network against the twin; on the four
             data sets with four-hour bars also the hybrid against two runs of the network (runs k and k+1) and
             against the average with the Transformer.
             Reading: the text says how many verdicts differ under the paired test and names each one, in either
             direction. The verdicts of the tables stay those of the rule fixed in the earlier plans.

  2 qlike    QLIKE under other choices for the level constant and for the bar of 10 October 2025, development data,
             four-hour bars, five runs, the hybrid against HAR, LightGBM and its network with the two tests:
               paper      the constant of the paper (mean of the ratio on the validation block, winsorised at the
                          99th percentile) and the cap on the squared range; must reproduce the numbers of the paper;
               mean       the mean of the ratio without winsorisation, with the cap;
               lognormal  exp(2 m + 2 s^2), with m and s^2 the mean and the variance of the errors of the log range
                          on the validation block (the mean of exp(2 e) for a normal error e), with the cap;
               without    the constant of the paper, no cap, and the test loss without the bar, the bar before it
                          and the 43 bars after it (the sample of the MSE check of the paper).
             Reading: the statement of the paper is that in QLIKE the hybrid is better than HAR and does not differ
             from LightGBM on these data. The text says whether both verdicts hold under each variant.

  3 limits   The range limits, development data, four-hour bars.
             (a) LightGBM fitted to the quantile is evaluated in the paper as it is, while the other limits are
                 shifted by a constant from the validation block. Here its forecast receives the same shift (the
                 level quantile of y minus the quantile forecast on the validation block of the coin and fold), at
                 95% and 99%, and the hybrid is compared with it in the tick loss by the two tests.
                 Reading: the sentence that a forecast of the mean with a constant is the stronger construction is
                 kept only if the hybrid is better than this shifted quantile fit in both tests at 95%; otherwise it
                 is replaced by what is found.
             (b) The tests of unconditional and conditional coverage per coin at 99% and 99.5%, for HAR, LightGBM,
                 the network and the hybrid (number of the 15 coins that reject at the 5% level, mean over the runs).
             (c) The constant as the ceil((n + 1) level)-th smallest validation error (the finite-sample quantile
                 of split-conformal prediction) in the place of the interpolated empirical quantile: exceedance
                 rates and tick losses at the three levels for the same four models, and the hybrid against LightGBM.
                 Reading: (b) and (c) are reported in one sentence. The statement that the use of the hybrid for
                 limits is a matter of the 95% level is kept unless the hybrid is better than LightGBM in both tests
                 at a higher level under (c).

  4 volume   The 15 new coins were ranked by their quote volume in 2024, which lies inside the earlier test period
             (30 October 2023 to 22 April 2025). Here the eligible candidates of the rule
             (configs/paper/v4/newcoins.json, everything else unchanged) are ranked by their quote volume in the 365
             days before that period (30 October 2022 to 29 October 2023; daily bars of the public interface, a new
             download of these bars only). Output: the 15 largest by that ranking and how many of them are among the
             15 new coins. Descriptive: no model is fitted on another set of coins.
             Reading: the text states the overlap next to the sentence on the year of the ranking.

  5 twinctl  The real twin of the paper has no projection between the encoder and the attention layer and keeps the
             default initialisation of its library, where the quaternion network has a 4H -> H projection and a
             normal Glorot initialisation with zero biases; the comparison of the paper does not separate the
             algebra from these two differences. Two further twins, trained on the development data (four-hour
             bars, the inputs, folds, protocol and seeds 0-4 of the network; configs/paper/v5/twinctl_pool_4h.yaml):
               hlp_gattn   the twin with a learned H -> H projection per group before the attention layer
                           (H = 34, 19,960 parameters);
               hlpg_gattn  the same with the initialisation of the quaternion layers in the recurrent weights
                           (normal with variance 2 / (fan_in + fan_out), zero biases, forget-gate bias as before).
             Comparisons with the two tests: the H-QLSTM against each of the two; each of the two against the twin
             of the paper; the hybrid (H-QLSTM + LightGBM) against the hybrid with each of the two, run k with run k;
             the hybrid with each against LightGBM and against its own network.
             Reading, fixed: the paper states that a real network of the same design and size gives a hybrid that is
             not worse, so that the gain is not owed to the algebra. The statement holds with the two differences
             removed if the hybrid with hlpg_gattn is not worse than the quaternion hybrid by the two-test rule
             (not different, or better). If the quaternion hybrid is better than it in both tests, the text says
             that the result with the twin of the paper depends on the projection or the initialisation, and the
             sentence of the abstract is reworded. For the networks alone the difference is reported as it is.

    .venv/bin/python -W ignore experiments/paper_v5_round5.py paired      tables/v5_r5_paired.csv
    .venv/bin/python -W ignore experiments/paper_v5_round5.py qlike       tables/v5_r5_qlike.csv
    .venv/bin/python -W ignore experiments/paper_v5_round5.py limits      tables/v5_r5_limits.csv
    .venv/bin/python -W ignore experiments/paper_v5_round5.py volume      tables/v5_r5_volume.csv
    (networks of check 5: configs/paper/v5/twinctl_pool_4h.yaml on Colab -> results/v5twin/out)
    .venv/bin/python -W ignore experiments/paper_v5_round5.py twinctl     tables/v5_r5_twinctl.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_paper as P3  # noqa: E402
import paper_v3_runs as RU  # noqa: E402
import paper_v4_round2 as R2  # noqa: E402
import paper_v4_round4 as R4  # noqa: E402
import paper_v5_stats as S5  # noqa: E402
from paper_v3_tests import verdict  # noqa: E402

CSV, RUNS, TRF, LGBM = RU.OUT, RU.RUNS, "Transformer", P3.LGBM
Q, R, G, HQ, HR = "H-QLSTM", "H-LSTM", "LightGBM", "H-QLSTM + LightGBM", "H-LSTM + LightGBM"
mix, sd = S5.mix, S5.sd
BAR = pd.Timestamp("2025-10-10 20:00")
pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 200)


def _out(rows: list, name: str) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, name), index=False)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================================ 1 paired
def _paired(B: dict, a: str, b: str) -> dict:
    r = R4._cmp(B, a, b)
    xa, xb = np.array([x.mse.mean() for x in B[a]]), np.array([x.mse.mean() for x in B[b]])
    assert len(xa) == len(xb) == len(RUNS), (a, b)
    p = float(stats.ttest_rel(xa, xb).pvalue)
    return dict(a=a, b=b, a_mean=r["a_mean"], b_mean=r["b_mean"], diff_pct=r["diff_pct"], corr_runs=float(np.corrcoef(xa, xb)[0, 1]),
                p_runs=r["p_runs"], p_runs_paired=p, p_time=r["p_time"], verdict=r["verdict"], verdict_paired=verdict(xa.mean() - xb.mean(), p, r["p_time"]))


def paired():
    import paper_v2_round2 as R0
    import paper_v3_bars as B3
    pairs = [(HQ, G), (HQ, Q), (HQ, R), (HQ, HR), (Q, R)]
    rows = []
    for tag in S5.SETS:
        M, c = S5._base(tag)
        X = {G: M[G], Q: M[Q], R: M[R], HQ: {k: mix(M[Q][k], M[G][k]) for k in RUNS}, HR: {k: mix(M[R][k], M[G][k]) for k in RUNS},
             "Two runs of H-QLSTM": {k: mix(M[Q][k], M[Q][(k + 1) % 5]) for k in RUNS},
             "H-QLSTM + LightGBM + Transformer": {k: mix(M[Q][k], M[G][k], M[TRF][k]) for k in RUNS}}
        B = R4._bars(X, c)
        for a, b in pairs + [(HQ, "Two runs of H-QLSTM"), (HQ, "H-QLSTM + LightGBM + Transformer")]:
            rows.append(dict(data=tag, freq="4h", **_paired(B, a, b)))
    for freq in ("1h", "1d"):
        c = R0.train_caps(freq)
        runs = {name: RU.forecasts(key, pats) for name, (key, pats) in B3.SOURCES[freq].items()}
        X = {G: {k: runs[LGBM][k] for k in RUNS}, Q: {k: runs[P3.NET][k] for k in RUNS}, R: {k: runs[P3.TWIN][k] for k in RUNS}}
        X[HQ], X[HR] = {k: mix(X[Q][k], X[G][k]) for k in RUNS}, {k: mix(X[R][k], X[G][k]) for k in RUNS}
        B = {n: [B3.bar_losses(f, c) for f in r.values()] for n, r in X.items()}
        for a, b in pairs:
            rows.append(dict(data="design", freq=freq, **_paired(B, a, b)))
    t = _out(rows, "v5_r5_paired.csv")
    ch = t[t.verdict != t.verdict_paired]
    print(f"\nverdicts that differ with the paired test: {len(ch)} of {len(t)}")
    print(ch[["data", "freq", "a", "b", "diff_pct", "p_runs", "p_runs_paired", "p_time", "verdict", "verdict_paired"]].round(4).to_string(index=False))


# ============================================================================================ 2 qlike
def _qlike(frame: pd.DataFrame, caps: pd.Series | None, kappa: str) -> pd.Series:
    """QLIKE of every test forecast of one run, indexed by (time, dataset), with the level constant `kappa`."""
    e = frame.reset_index()
    e = e.assign(s2=np.exp(2 * e.y), raw=np.exp(2 * e.pred), err=e.y - e.pred)
    if caps is not None:
        e = e.join(caps, on=["dataset", "fold"])
        e["s2"] = np.minimum(e.s2, e.cap)
    by = ["dataset", "fold"]
    val = e[e.split == "val"].assign(u=lambda d: d.s2 / d.raw)
    if kappa == "paper":
        top = val.groupby(by).u.transform(lambda x: x.quantile(0.99))
        k = val.assign(u=np.minimum(val.u, top)).groupby(by).u.mean()
    elif kappa == "mean":
        k = val.groupby(by).u.mean()
    elif kappa == "lognormal":
        k = np.exp(2 * val.groupby(by).err.mean() + 2 * val.groupby(by).err.var(ddof=1))
    else:
        raise ValueError(kappa)
    test = e[e.split == "test"].join(k.rename("k"), on=by)
    h = test.k * test.raw
    return pd.Series((test.s2 / h - np.log(test.s2 / h) - 1).to_numpy(), index=pd.MultiIndex.from_arrays([test.time, test.dataset], names=["time", "dataset"])).sort_index()


def qlike():
    M, c = S5._base("design")
    X = {"HAR": M["HAR"], G: M[G], Q: M[Q], HQ: {k: mix(M[Q][k], M[G][k]) for k in RUNS}}
    ref = R4._bars({HQ: X[HQ]}, c)[HQ]                                   # the QLIKE of the paper
    t = R4._time(ref[0].index)
    bar = BAR.tz_localize(t.tz)
    lo, hi = bar - pd.Timedelta(hours=4), bar + pd.Timedelta(hours=4 * 43)
    keep = lambda idx: np.asarray((R4._time(idx) < lo) | (R4._time(idx) > hi))
    rows = []
    for name, caps, kappa, sel in (("paper", c, "paper", None), ("mean", c, "mean", None), ("lognormal", c, "lognormal", None), ("without", None, "paper", keep)):
        B = {n: [pd.DataFrame({"qlike": _qlike(f, caps, kappa)}) for f in r.values()] for n, r in X.items()}
        if name == "paper":
            assert all(np.allclose(a.qlike.to_numpy(), b.qlike.to_numpy()) for a, b in zip(B[HQ], ref)), "the routine does not reproduce the QLIKE of the paper"
        for o in ("HAR", G, Q):
            r = R4._cmp(B, HQ, o, loss="qlike", keep=sel)
            rows.append(dict(variant=name, **{k: r[k] for k in ("a", "b", "a_mean", "a_sd", "b_mean", "b_sd", "diff_pct", "lo90", "hi90", "p_runs", "p_time", "coins_better", "bars", "verdict")}))
    _out(rows, "v5_r5_qlike.csv")


# ============================================================================================ 3 limits
def _shifted(frames: dict, level: float, finite: bool = False) -> list:
    """List over runs of the test frame with the limit forecast + constant; the constant is the level quantile of the
    validation errors of the coin and fold (interpolated, as in the paper, or the finite-sample order statistic)."""
    def quant(x: pd.Series) -> float:
        if not finite:
            return float(x.quantile(level))
        v = np.sort(x.to_numpy())
        return float(v[min(int(np.ceil((len(v) + 1) * level)), len(v)) - 1])
    runs = []
    for f in frames.values():
        d = f.reset_index()
        val, test = d[d.split == "val"], d[d.split == "test"].copy()
        cst = (val.y - val.pred).groupby([val.dataset, val.fold]).apply(quant).rename("c")
        test = test.join(cst, on=["dataset", "fold"])
        u = test.y - (test.pred + test.c)
        test["hit"], test["tick"] = (u > 0).astype(int), u * (level - (u < 0))
        runs.append(test.sort_values(["dataset", "time"]).reset_index(drop=True))
    return runs


def _stat(runs: list) -> dict:
    bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in runs]
    return dict(tick=np.array([x.tick.mean() for x in runs]), bar=sum(bars) / len(bars), exceed=float(np.mean([x.hit.mean() for x in runs])))


def limits():
    import paper_v4_round3 as R3
    M = S5.M0()
    names = {"HAR": "HAR", G: LGBM, Q: P3.NET, HQ: P3.HYBRID}
    rows = []
    for level in (0.95, 0.99, 0.995):
        base = {n: _shifted(M[key], level) for n, key in names.items()}
        fin = {n: _shifted(M[key], level, finite=True) for n, key in names.items()}
        check = R2._limits(M, [P3.HYBRID], rate=1 - level)[P3.HYBRID]
        assert all(np.allclose(a.tick.to_numpy(), b.tick.to_numpy()) for a, b in zip(base[HQ], check)), "the routine does not reproduce the limits of the paper"
        for kind, L in (("paper", base), ("finite-sample quantile", fin)):
            S = {n: _stat(r) for n, r in L.items()}
            for n, runs in L.items():
                ps = [np.array([R2._lr_tests(g.hit.to_numpy(), p=1 - level) for _, g in x.groupby("dataset")]) for x in runs]
                r = dict(level=level, constant=kind, model=n, exceed=S[n]["exceed"], tick_mean=S[n]["tick"].mean(), tick_har=S[n]["tick"].mean() / S["HAR"]["tick"].mean(),
                         coins_reject_uc=float(np.mean([(p[:, 0] < 0.05).sum() for p in ps])), coins_reject_cc=float(np.mean([(p[:, 2] < 0.05).sum() for p in ps])))
                if n != HQ:
                    r.update(R2._two_tests(S[HQ]["tick"], S[n]["tick"], S[HQ]["bar"], S[n]["bar"]))
                rows.append(r)
        if level in R3.QUANTILES:
            q = RU.forecasts(("PM", "lgbmq-pool"), [f"{R3.EXTRA_DIR}/lgbmq{int(round(100 * level))}"])
            assert sorted(q) == list(RUNS)
            for kind, finite in (("paper", False), ("finite-sample quantile", True)):
                s, h = _stat(_shifted(q, level, finite)), _stat((fin if finite else base)[HQ])
                har = _stat((fin if finite else base)["HAR"])["tick"].mean()
                rows.append(dict(level=level, constant=kind, model="LightGBM, quantile objective, shifted", exceed=s["exceed"], tick_mean=s["tick"].mean(),
                                 tick_har=s["tick"].mean() / har, **R2._two_tests(h["tick"], s["tick"], h["bar"], s["bar"])))
    _out(rows, "v5_r5_limits.csv")


# ============================================================================================ 4 volume
def volume():
    import paper_v4_newcoins as N
    from urllib.parse import quote
    u = json.load(open(N.UNIVERSE))
    start, end = pd.Timestamp("2022-10-30", tz="UTC"), pd.Timestamp("2023-10-29", tz="UTC")
    rows = []
    for r in u["ranking"]:
        if not r["eligible"]:
            continue
        part = N._get(f"{N.API}/klines?symbol={quote(r['coin'].upper() + 'USDT')}&interval=1d&startTime={int(start.timestamp() * 1000)}"
                      f"&endTime={int(end.timestamp() * 1000)}&limit=1000")
        rows.append(dict(coin=r["coin"], quote_volume_2024=r["quote_volume_2024"], quote_volume_before=float(sum(float(x[7]) for x in part)), days_before=len(part)))
    t = pd.DataFrame(rows)
    t["rank_2024"] = t.quote_volume_2024.rank(ascending=False).astype(int)
    t["rank_before"] = t.quote_volume_before.rank(ascending=False).astype(int)
    t["chosen"] = t.coin.isin(u["chosen"])
    assert set(t[t.rank_2024 <= 15].coin) == set(u["chosen"]), "the ranking of the plan is not reproduced"
    t = t.sort_values("rank_before")
    t.to_csv(os.path.join(CSV, "v5_r5_volume.csv"), index=False)
    print(t.head(25).to_string(index=False))
    top = t[t.rank_before <= 15]
    print(f"\neligible candidates: {len(t)}; of the 15 largest by the volume of {start.date()} to {end.date()}, {int(top.chosen.sum())} are among the 15 new coins")
    print("new coins outside these 15:", t[t.chosen & (t.rank_before > 15)][["coin", "rank_before"]].to_dict("records"))
    print("in their place:", list(top[~top.chosen].coin))


# ============================================================================================ 5 twinctl
TWIN_RUNS = "results/v5twin/out/v5_twinctl_pool_4h_*"


def twinctl():
    M, c = S5._base("design")
    ref = M[R][0].index
    X = {G: M[G], Q: M[Q], R: M[R], HQ: {k: mix(M[Q][k], M[G][k]) for k in RUNS}, HR: {k: mix(M[R][k], M[G][k]) for k in RUNS}}
    new = {"H-LSTM with projection": "hlp_gattn-pool", "H-LSTM with projection and initialisation": "hlpg_gattn-pool"}
    for label, key in new.items():
        F = RU.forecasts(("PRMC", key), [TWIN_RUNS])
        assert sorted(F) == list(RUNS), (label, sorted(F))
        X[label] = {k: F[k].reindex(ref) for k in RUNS}
        assert all(np.isfinite(f.pred).all() and np.allclose(f.y, M[R][0].y) for f in X[label].values()), label
        X[f"{label} + LightGBM"] = {k: mix(X[label][k], M[G][k]) for k in RUNS}
    B = R4._bars(X, c)
    rows = []
    for label in new:
        for a, b in ((Q, label), (label, R), (HQ, f"{label} + LightGBM"), (f"{label} + LightGBM", HR), (f"{label} + LightGBM", G), (f"{label} + LightGBM", label)):
            r = R4._cmp(B, a, b)
            xa, xb = np.array([x.mse.mean() for x in B[a]]), np.array([x.mse.mean() for x in B[b]])
            rows.append(dict(**r, p_runs_paired=float(stats.ttest_rel(xa, xb).pvalue)))
    _out(rows, "v5_r5_twinctl.csv")


if __name__ == "__main__":
    {"paired": paired, "qlike": qlike, "limits": limits, "volume": volume, "twinctl": twinctl}[sys.argv[1]]()
