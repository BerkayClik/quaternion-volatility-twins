"""
The hybrid on hourly and daily bars (robustness check for the manuscript, 2026-11-22). Fixed before any of these
numbers is computed. Known before: three-seed averages of the parts on these bar lengths (tables/v3_grid_bars.csv:
hourly H-QLSTM 0.870 x HAR, its real twin 0.867, lagged LightGBM-S 0.879; daily 0.869, 0.869, 0.860). Not computed
before: any hybrid on these bar lengths, any QLIKE of these models on these bar lengths, any single run.

  system     as on 4-hour bars, nothing tuned: hq_gattn on price with range, market and calendar (20,000
             parameters; windows of 48 hourly and 30 daily bars, the windows of the paper), its real twin hl_gattn,
             the lagged LightGBM-S of the paper for the bar length (HAR lags 1/24/168 and 1/7/30), the plain average.
  runs       seeds 0-4. Seeds 0-2 exist (v3_a_pool_1h, v3_a_pool_1d; results/r8_local); seeds 3-4 are new
             (configs/paper/v3/a_seeds_pool_1h.yaml, a_seeds_pool_1d.yaml; `lgbm` below).
  models     HAR, the seasonal HAR (HAR-S x; HAR-S on daily bars, where the two coincide), the lagged LightGBM-S,
             H-LSTM, H-QLSTM, H-LSTM + LightGBM and H-QLSTM + LightGBM (run k with run k).
  losses     MSE and QLIKE of a run as in paper_v3_runs (QLIKE with the cap of train_caps for the bar length);
             mean and standard deviation over the runs. The 4-hour rows are computed by the same code and must
             equal tables/v3_runs.csv and v3_tests.csv.
  tests      the hybrid against every other model, as in paper_v3_tests: Welch's t-test over the runs (one-sample
             against the two HAR models) and the Diebold-Mariano test over time on the loss difference of every
             bar, averaged over runs and coins. Holm per bar length, loss and test (in the CSV file).
  rule       as in paper_v3_tests: better (worse) only if the mean is lower (higher) and both tests give p < 0.05.
             The result of the 4-hour bars, "the hybrid has a lower MSE than LightGBM and than the H-QLSTM", counts
             as confirmed on a bar length only if it holds there by this rule against both.

    .venv/bin/python -W ignore experiments/paper_v3_bars.py lgbm    seeds 3-4 of the lagged LightGBM-S, 1h and 1d
    .venv/bin/python -W ignore experiments/paper_v3_bars.py         tables/v3_bars.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_runs as RU  # noqa: E402

OUT, RUNS = RU.OUT, RU.RUNS
NET, TWIN, LGBM = ("PRMC", "hq_gattn-pool"), ("PRMC", "hl_gattn-pool"), ("PM", "lgbmsl-pool")
HYBRID, TWIN_HYBRID = "H-QLSTM + LightGBM", "H-LSTM + LightGBM"
V3 = RU.V3
# bar length -> model -> (key, run patterns)
SOURCES = {
    "1h": {"HAR": (("P", "har"), ["results/final_fb/out/final_single_1h_*"]),
           "HAR-S x": (("P", "hars_x"), ["results/r8_local/final_harsx_1h_r8"]),
           "LightGBM-S, lagged": (LGBM, ["results/r8_local/final_lgbmlags_1h_r8", RU.LGBM_NEW.format(f="1h")]),
           "H-LSTM (real twin)": (TWIN, [V3.format("pool_1h"), V3.format("seeds_pool_1h")]),
           "H-QLSTM": (NET, [V3.format("pool_1h"), V3.format("seeds_pool_1h")])},
    "4h": {"HAR": RU.FIXED["HAR"], "HAR-S x": RU.FIXED["HAR-S x"], "LightGBM-S, lagged": RU.SEEDED["LightGBM-S, lagged"],
           "H-LSTM (real twin)": RU.SEEDED["H-LSTM (real twin)"], "H-QLSTM": RU.SEEDED["H-QLSTM (candidate A)"]},
    "1d": {"HAR": (("P", "har"), ["results/final_fa/out/final_single_1d_*"]),
           "HAR-S x": (("PMSE", "hars"), ["results/final_local/final_har_season_1d_*"]),
           "LightGBM-S, lagged": (LGBM, ["results/r8_local/final_lgbmlags_1d_r8", RU.LGBM_NEW.format(f="1d")]),
           "H-LSTM (real twin)": (TWIN, [V3.format("pool_1d"), V3.format("seeds_pool_1d")]),
           "H-QLSTM": (NET, [V3.format("pool_1d"), V3.format("seeds_pool_1d")])},
}
ORDER = ["HAR", "HAR-S x", "LightGBM-S, lagged", "H-LSTM (real twin)", "H-QLSTM", TWIN_HYBRID, HYBRID]


def lgbm():
    """Seeds 3 and 4 of the lagged LightGBM-S on hourly and daily bars, fitted exactly as seeds 0-2."""
    import paper_round8_checks as R8
    R8.LGBM_LAG_DIR = RU.LGBM_NEW
    R8.lgbm_save(freqs=("1h", "1d"), seeds=(3, 4))


def bar_losses(frame: pd.DataFrame, caps: pd.Series) -> pd.DataFrame:
    """Squared error and QLIKE of every test forecast of one run, indexed by (time, dataset)."""
    import paper_v2_round2 as R
    ens = frame.reset_index().assign(groupset="g", model="m")
    test = R._calibrated(ens, None, caps=caps)
    return pd.DataFrame({"mse": (test.pred - test.y) ** 2, "qlike": test.ql}).set_index([test.time, test.dataset]).sort_index()


def main():
    import paper_v2_round2 as R
    from scipy import stats
    from src.paper.evaluate import diebold_mariano, holm
    from paper_v3_tests import verdict
    rows = []
    for freq, sources in SOURCES.items():
        caps = R.train_caps(freq)
        runs = {name: RU.forecasts(key, pats) for name, (key, pats) in sources.items()}
        seeds = [k for k in RUNS if all(k in runs[m] for m in ("LightGBM-S, lagged", "H-LSTM (real twin)", "H-QLSTM"))]
        if len(seeds) < len(RUNS):
            print(f"{freq}: runs {seeds} only", flush=True)
        if not seeds:
            continue
        mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
        models = {m: {0: runs[m][0]} for m in ("HAR", "HAR-S x")}
        models.update({m: {k: runs[m][k] for k in seeds} for m in ("LightGBM-S, lagged", "H-LSTM (real twin)", "H-QLSTM")})
        models[TWIN_HYBRID] = {k: mix(runs["H-LSTM (real twin)"][k], runs["LightGBM-S, lagged"][k]) for k in seeds}
        models[HYBRID] = {k: mix(runs["H-QLSTM"][k], runs["LightGBM-S, lagged"][k]) for k in seeds}
        per_run, per_bar = {}, {}
        for name, by_run in models.items():
            bars = [bar_losses(f, caps) for f in by_run.values()]
            assert all(b.index.equals(bars[0].index) for b in bars), (freq, name)
            per_run[name] = pd.DataFrame([b.mean() for b in bars])
            per_bar[name] = sum(bars) / len(bars)
        n = len(per_bar[HYBRID])
        assert all(len(b) == n and not b.isna().any().any() for b in per_bar.values()), freq
        block = []
        for name in ORDER:
            r = dict(bars=freq, model=name, runs=len(per_run[name]), test_coin_bars=n)
            for loss in ("mse", "qlike"):
                x = per_run[name][loss].to_numpy()
                har = per_run["HAR"][loss].mean()
                r.update({f"{loss}_mean": x.mean(), f"{loss}_sd": np.std(x, ddof=1) if len(x) > 1 else np.nan, f"{loss}_har": x.mean() / har})
                if name == HYBRID:
                    continue
                o = per_run[HYBRID][loss].to_numpy()          # the hybrid against this model
                if len(x) > 1:
                    t_stat, p_t = stats.ttest_ind(o, x, equal_var=False)
                else:
                    t_stat, p_t = stats.ttest_1samp(o, x[0])
                d = (per_bar[HYBRID][loss] - per_bar[name][loss]).unstack("dataset")
                assert not d.isna().any().any(), (freq, name)
                series = d.mean(axis=1).sort_index().to_numpy()
                dm, p_dm = diebold_mariano(series, np.zeros(len(series)))
                se = abs(series.mean() / dm)
                assert np.isclose(series.mean(), o.mean() - x.mean(), rtol=1e-6, atol=1e-9), (freq, name, loss)
                r.update({f"{loss}_diff_pct": 100 * (o.mean() / x.mean() - 1), f"{loss}_diff_lo": 100 * (series.mean() - 1.645 * se) / x.mean(),
                          f"{loss}_diff_hi": 100 * (series.mean() + 1.645 * se) / x.mean(), f"{loss}_coins_better": int((d.mean(axis=0) < 0).sum()),
                          f"{loss}_p_runs": p_t, f"{loss}_p_time": p_dm, f"{loss}_verdict": verdict(o.mean() - x.mean(), p_t, p_dm)})
            block.append(r)
        b = pd.DataFrame(block)
        tested = b.model != HYBRID
        for col in ("mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"):
            b.loc[tested, col + "_holm"] = holm(b.loc[tested, col].to_numpy())
        rows.append(b)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(os.path.join(OUT, "v3_bars.csv"), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 40)
    print(t[["bars", "model", "runs", "mse_mean", "mse_sd", "mse_har", "mse_diff_pct", "mse_p_runs", "mse_p_time", "mse_coins_better",
             "mse_verdict", "qlike_mean", "qlike_sd", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_verdict"]].round(4).to_string(index=False))


if __name__ == "__main__":
    {"lgbm": lgbm}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
