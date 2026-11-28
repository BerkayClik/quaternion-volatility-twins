"""
Confirmation of the dual-view hybrid on coins that played no part in its design (paper v4, 2026-11-23).

Every design choice of the paper (the network, the idea of the hybrid, the choice of real layers) was made with the
forecasts for 15 coins on four test blocks in view. This experiment runs the frozen system on 15 other coins. This
file, the option `exchange_only` of the data loader and the rule below were committed before any data of a new coin
were downloaded.

PLAN (fixed in advance)
  Coins      Binance spot pairs quoted in USDT that are trading on the day of the download, without the 15 coins of
             the paper and without the tokens of EXCLUDE (stablecoins and fiat tokens, tokenised gold, wrapped and
             staked versions of other coins, leveraged tokens), whose first daily bar is before 2021-07-01 and
             whose daily bars have no gap of more than 7 days up to 2026-10-14. Ranked by the quote volume of the
             calendar year 2024, which precedes the first test block; the 15 largest are the new coins. The
             selection uses listing dates and volumes only.
  Sample     four-hour bars from 2020-01-01 (or from the listing, if later), features from the exchange file alone
             (`exchange_only`), up to the end of the paper's sample. The four test blocks are those of the paper
             (135 days each, ending 2026-10-14 16:00 UTC; validation 180 days), 48,600 test forecasts if no bar is
             missing.
  System     unchanged, nothing tuned: hl_gattn and hq_gattn on price with range, market and calendar, 20,000
             parameters, window 42, the training protocol of the paper, one network for the 15 new coins, trained
             from scratch on their bars; the lagged LightGBM-S of the paper; their plain average. The Transformer
             of the paper as a third network. Seeds 0 to 4 (configs/paper/v4/new_pool_4h.yaml, written by
             `universe`). Benchmarks: HAR and HAR-S x per coin.
  Losses     MSE of the log range (primary); QLIKE with the cap and level constant of the paper; tick loss of the
             95% range limits as in paper_v3_paper.limits. Mean and standard deviation over five runs.
  Tests      the two tests and the rule of paper_v3_tests (Welch over runs, one-sample against the HAR models;
             Diebold-Mariano over time on the loss difference per bar averaged over runs and coins; better or worse
             only if both p < 0.05); Holm per hybrid, loss and test.
  Rule       PRIMARY: the result of the paper counts as confirmed on the new coins if the hybrid with real layers
             has a lower MSE than LightGBM and than the H-LSTM by the rule. Reported whatever it is.
             SECONDARY, reported as they are: the same for the quaternion version; the hybrid against HAR, HAR-S x
             and the Transformer; the two versions against each other; the pairs Transformer + LightGBM and
             H-LSTM + Transformer and the plain average of H-LSTM, LightGBM and the Transformer against the hybrid;
             the tick loss of the range limits of the hybrid against LightGBM and the H-LSTM.

    .venv/bin/python -W ignore experiments/paper_v4_newcoins.py universe   rule -> configs/paper/v4/newcoins.json and the config
    .venv/bin/python -W ignore experiments/paper_v4_newcoins.py download   four-hour bars of the new coins -> data/binance
    .venv/bin/python -W ignore experiments/paper_v4_newcoins.py save       benchmarks -> results/v4new/bench
    (networks: configs/paper/v4/new_pool_4h.yaml on Colab -> results/v4new/out)
    .venv/bin/python -W ignore experiments/paper_v4_newcoins.py eval       tables/v4_newcoins.csv, v4_newcoins_limits.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

API = "https://data-api.binance.vision/api/v3"
OLD = ["btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "ltc", "dot", "avax", "trx", "bch", "xlm", "atom"]
EXCLUDE = {  # not volatile crypto assets of their own
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "PYUSD", "USD1", "XUSD", "BFUSD", "USDE", "EUR", "EURI", "AEUR", "UST", "USTC", "SUSD",
    "PAXG", "XAUT", "WBTC", "WBETH", "BETH", "WETH", "BNSOL", "STETH", "WSTETH", "RETH", "CBETH", "LSETH",
}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")
N_NEW, LISTED_BEFORE, RANK_YEAR, MAX_GAP_DAYS = 15, "2021-07-01", 2024, 7
END = pd.Timestamp("2026-10-14 16:00", tz="UTC")
START, FREQ, WINDOW, SEEDS = "2020-01-01", "4h", 42, (0, 1, 2, 3, 4)
FOLDS = dict(n=4, test_days=135, val_days=180, end="2026-10-14 16:00")
UNIVERSE, CONFIG = "configs/paper/v4/newcoins.json", "configs/paper/v4/new_pool_4h.yaml"
BENCH, OUT_RUNS = "results/v4new/bench", "results/v4new/out/v4_new_pool_4h_*"
CSV = "docs/paper/manuscript_v2/tables"
TAG = "newcoins"            # name of the result tables; paper_v4_round3 reuses this module for other folds
HYB_R, HYB_Q, NET_R, NET_Q, LGBM, TRF = "H-LSTM + LightGBM", "H-QLSTM + LightGBM", "H-LSTM", "H-QLSTM", "LightGBM", "Transformer"


def _get(url: str):
    from urllib.request import Request, urlopen
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={"User-Agent": "klines-download/1.0"}), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 + 10 * attempt)
    raise RuntimeError(f"{url}: {err}")


def _daily(symbol: str, start_ms: int = 0) -> pd.DataFrame:
    rows = []
    while True:
        part = _get(f"{API}/klines?symbol={symbol}&interval=1d&startTime={start_ms}&limit=1000")
        rows += part
        if len(part) < 1000:
            break
        start_ms = part[-1][0] + 1
        time.sleep(0.15)
    d = pd.DataFrame([(r[0], float(r[7])) for r in rows], columns=["t", "quote_volume"])
    d.index = pd.to_datetime(d.pop("t"), unit="ms", utc=True)
    return d


def universe():
    """Apply the rule of the plan and write the list of new coins and the config of the networks."""
    info = _get(f"{API}/exchangeInfo")
    cand = sorted(s["baseAsset"] for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s.get("isSpotTradingAllowed", True))
    cand = [b for b in cand if b.lower() not in OLD and b not in EXCLUDE and not b.endswith(LEVERAGED)]
    # the listing date and the daily bars of every candidate, kept in a cache so that an interrupted run continues
    # (added after the first run was stopped by a time limit; the rule is unchanged)
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote
    cache_path = "results/v4new/universe_cache.json"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def look(base: str):
        symbol = quote(f"{base}USDT")                      # a few recent listings have non-ASCII names
        first = _get(f"{API}/klines?symbol={symbol}&interval=1d&startTime=0&limit=1")
        if not first or pd.Timestamp(first[0][0], unit="ms", tz="UTC") >= pd.Timestamp(LISTED_BEFORE, tz="UTC"):
            return base, None
        d = _daily(symbol)
        d = d[d.index <= END]
        gap = d.index.to_series().diff().max()
        ok = len(d) > 0 and d.index[-1] >= END.normalize() - pd.Timedelta(days=1) and gap <= pd.Timedelta(days=MAX_GAP_DAYS)
        return base, dict(coin=base.lower(), first_bar=str(d.index[0].date()), last_bar=str(d.index[-1].date()), max_gap_days=int(gap.days),
                          quote_volume_2024=float(d[d.index.year == RANK_YEAR].quote_volume.sum()), eligible=bool(ok))

    todo = [b for b in cand if b not in cache]
    with ThreadPoolExecutor(6) as pool:
        for i, (base, row) in enumerate(pool.map(look, todo)):
            cache[base] = row
            if i % 20 == 0 or i == len(todo) - 1:
                json.dump(cache, open(cache_path, "w"))
                print(f"{len(cache)}/{len(cand)} candidates looked up", flush=True)
    rows = [cache[b] for b in cand if cache[b] is not None]
    t = pd.DataFrame(rows).sort_values("quote_volume_2024", ascending=False)
    chosen = t[t.eligible].head(N_NEW).coin.tolist()
    os.makedirs(os.path.dirname(UNIVERSE), exist_ok=True)
    json.dump(dict(rule=dict(listed_before=LISTED_BEFORE, rank_year=RANK_YEAR, max_gap_days=MAX_GAP_DAYS, n=N_NEW, excluded=sorted(EXCLUDE),
                             date=str(pd.Timestamp.now(tz="UTC").date()), candidates=len(cand)),
                   chosen=chosen, ranking=t.to_dict("records")), open(UNIVERSE, "w"), indent=1)
    with open(CONFIG, "w") as f:
        f.write(f"""# V4 confirmation on 15 coins that played no part in the design (plan in experiments/paper_v4_newcoins.py, committed
# before any data of these coins were downloaded; the coins follow from the rule there, see newcoins.json).
# The system of the paper, unchanged: both versions of the network and the Transformer, five seeds, the folds of the paper.
name: v4_new_pool_4h
datasets: [{", ".join(c + "_4h" for c in chosen)}]
tasks: [range]
exchange_only: true
start: "{START}"
groupsets:
  PRMC: [price_r, market, calendar]
models: []
pooled_models: [hq_gattn, hl_gattn, transformer]
seeds: [0, 1, 2, 3, 4]
window: {{4h: 42}}
folds: {{n: 4, test_days: 135, val_days: 180, end: "2026-10-14 16:00"}}
budget: 20000
train: {{lr: 0.001, batch_size: 128, max_epochs: 150, patience: 10, grad_clip: 1.0}}
""")
    print(t.head(25).to_string(index=False))
    print("chosen:", chosen)


def chosen() -> list[str]:
    return json.load(open(UNIVERSE))["chosen"]


def download():
    """Four-hour bars of the new coins with the downloader of the paper."""
    sys.path.insert(0, "scripts")
    import download_binance as D
    for coin in chosen():
        start_ms = int(pd.Timestamp(D.START, tz="UTC").timestamp() * 1000)
        df = pd.DataFrame(D.fetch_klines(f"{coin.upper()}USDT", FREQ, start_ms), columns=D.COLUMNS)
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.index.name = "datetime"
        df = df[D.KEEP].astype(float)
        df = df[df.index <= END + pd.Timedelta(hours=4)]          # the bar after the last row is the last target
        df.to_csv(f"data/binance/{coin}_{FREQ}.csv")
        print(coin, len(df), df.index[0], df.index[-1], flush=True)


def frames() -> dict:
    """Datasets and folds exactly as the runner builds them from the config."""
    from src.paper.data import make_dataset, walk_forward_folds
    out = {}
    for coin in chosen():
        d = make_dataset(coin, FREQ, start=START, exchange_only=True)
        out[f"{coin}_{FREQ}"] = (d, walk_forward_folds(d.frame.index, FOLDS["n"], FOLDS["test_days"], FOLDS["val_days"], end=FOLDS["end"]))
    return out


def _save(name: str, ds: str, gs: str, seed: int, fold: int, frame: pd.DataFrame, res: dict):
    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
    t = idx.as_unit("ns").asi8
    y = frame["y_range"].to_numpy(float)
    np.savez(os.path.join(BENCH, "preds", f"{ds}__range__{gs}__{name}__s{seed}__f{fold}.npz"),
             test_time=t[res["test_rows"]], test_y=y[res["test_rows"]].astype("float32"), test_pred=np.asarray(res["test_pred"], "float32"),
             val_time=t[res["val_rows"]], val_y=y[res["val_rows"]].astype("float32"), val_pred=np.asarray(res["val_pred"], "float32"),
             meta=np.array(json.dumps(dict(freq=FREQ, seed=seed, fold=fold, model=name))))


def save():
    """Benchmarks on the new coins, fitted as in the paper: HAR and HAR-S x per coin by OLS on the training block, the
    lagged LightGBM-S pooled over the coins (paper_round8_checks.lgbm_save)."""
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _har_design, _ols, predict_baseline, predict_baseline_pooled
    os.makedirs(os.path.join(BENCH, "preds"), exist_ok=True)
    json.dump({"config": {"name": "v4_new_bench", "note": "paper_v4_newcoins.save"}}, open(os.path.join(BENCH, "config.json"), "w"), indent=1)
    fr = frames()
    for ds, (d, folds) in fr.items():
        y = d.frame["y_range"].to_numpy(float)
        D = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[FREQ]), _interactions(d.frame.index, d.step)])
        for k, fold in enumerate(folds):
            _save("har", ds, "P", 0, k, d.frame, predict_baseline("har", d.frame, "range", fold, WINDOW, FREQ))
            ok = fold["train"][~np.isnan(D[fold["train"]]).any(1)]
            b = _ols(D[ok], y[ok])
            te, va = (r[r >= WINDOW - 1] for r in (fold["test"], fold["val"]))
            _save("hars_x", ds, "P", 0, k, d.frame, dict(test_rows=te, test_pred=b[0] + D[te] @ b[1:], val_rows=va, val_pred=b[0] + D[va] @ b[1:]))
    dss = list(fr)
    Xs = [fr[ds][0].X(["price", "market"]) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]]) for X in Xs]
    for seed in SEEDS:
        for k in range(FOLDS["n"]):
            res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range", [fr[ds][1][k] for ds in dss], WINDOW, FREQ, Xl, seed=seed)
            for ds, r in zip(dss, res):
                _save("lgbmsl-pool", ds, "PM", seed, k, fr[ds][0].frame, r)
            print("lgbmsl-pool seed", seed, "fold", k, flush=True)
    print("saved", BENCH)


def evaluate():
    import paper_v3_runs as RU
    from paper_v3_paper import compare
    from paper_v3_tests import bar_losses, verdict
    from scipy import stats as st
    from src.paper.evaluate import diebold_mariano, holm
    fr = frames()
    caps = pd.DataFrame([dict(dataset=ds, fold=k, cap=float(np.exp(2 * np.nanmax(d.frame["y_range"].to_numpy(float)[fo["train"]]))))
                         for ds, (d, folds) in fr.items() for k, fo in enumerate(folds)]).set_index(["dataset", "fold"]).cap
    get = RU.forecasts
    M = {"HAR": {0: get(("P", "har"), [BENCH])[0]}, "HAR-S x": {0: get(("P", "hars_x"), [BENCH])[0]}, LGBM: get(("PM", "lgbmsl-pool"), [BENCH]),
         TRF: get(("PRMC", "transformer-pool"), [OUT_RUNS]), NET_Q: get(("PRMC", "hq_gattn-pool"), [OUT_RUNS]), NET_R: get(("PRMC", "hl_gattn-pool"), [OUT_RUNS])}
    for name in (LGBM, TRF, NET_Q, NET_R):
        assert sorted(M[name]) == list(SEEDS), (name, sorted(M[name]))
    mix = lambda *fs: fs[0].assign(pred=sum(f.reindex(fs[0].index).pred for f in fs) / len(fs))
    M[HYB_Q] = {k: mix(M[NET_Q][k], M[LGBM][k]) for k in SEEDS}
    M[HYB_R] = {k: mix(M[NET_R][k], M[LGBM][k]) for k in SEEDS}
    M["Transformer + LightGBM"] = {k: mix(M[TRF][k], M[LGBM][k]) for k in SEEDS}
    M["H-LSTM + Transformer"] = {k: mix(M[NET_R][k], M[TRF][k]) for k in SEEDS}
    M["H-LSTM + LightGBM + Transformer"] = {k: mix(M[NET_R][k], M[LGBM][k], M[TRF][k]) for k in SEEDS}
    per_run, per_bar = {}, {}
    for name, by_run in M.items():
        bars = [bar_losses(f, caps) for f in by_run.values()]
        assert all(b.index.equals(bars[0].index) for b in bars), name
        per_run[name], per_bar[name] = pd.DataFrame([b.mean() for b in bars]), sum(bars) / len(bars)
    n = len(per_bar[HYB_R])
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    har = per_run["HAR"].mean()
    rows = []
    for ours in (HYB_R, HYB_Q):
        for name in M:
            p = per_run[name]
            r = dict(ours=ours, model=name, runs=len(p), test_coin_bars=n, mse_mean=p.mse.mean(), mse_sd=sd(p.mse), mse_har=p.mse.mean() / har.mse,
                     qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike))
            if name != ours:
                r.update(compare(per_run, per_bar, ours, name))
            rows.append(r)
    t = pd.DataFrame(rows)
    for _, g in t.groupby("ours"):
        for col in ("mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"):
            ok = g.index[g[col].notna()]
            t.loc[ok, col + "_holm"] = holm(t.loc[ok, col].to_numpy())
    t.to_csv(os.path.join(CSV, f"v4_{TAG}.csv"), index=False)

    # range limits (paper_v3_paper.limits)
    rate, lim_run, lim_bar = 0.05, {}, {}
    for name in ("HAR", "HAR-S x", LGBM, TRF, NET_Q, NET_R, HYB_Q, HYB_R):
        runs, bars = [], []
        for f in M[name].values():
            d = f.reset_index()
            val, test = d[d.split == "val"], d[d.split == "test"].copy()
            c = (val.y - val.pred).groupby([val.dataset, val.fold]).quantile(1 - rate).rename("c")
            test = test.join(c, on=["dataset", "fold"])
            u = test.y - (test.pred + test.c)
            test["tick"] = u * ((1 - rate) - (u < 0))
            runs.append(dict(exceed=float((u > 0).mean()), tick=float(test.tick.mean())))
            bars.append(test.set_index(["time", "dataset"]).tick.sort_index())
        lim_run[name], lim_bar[name] = runs, sum(bars) / len(bars)
    rows = []
    for ours in (HYB_R, HYB_Q):
        o = np.array([x["tick"] for x in lim_run[ours]])
        for name, r in lim_run.items():
            tick, ex = np.array([x["tick"] for x in r]), np.array([x["exceed"] for x in r])
            row = dict(ours=ours, model=name, exceed_mean=ex.mean(), tick_mean=tick.mean(), tick_sd=sd(tick), tick_har=tick.mean() / lim_run["HAR"][0]["tick"])
            if name != ours:
                p_t = st.ttest_ind(o, tick, equal_var=False).pvalue if len(tick) > 1 else st.ttest_1samp(o, tick[0]).pvalue
                d = (lim_bar[ours] - lim_bar[name]).unstack("dataset")
                series = d.mean(axis=1).sort_index().to_numpy()
                _, p_dm = diebold_mariano(series, np.zeros(len(series)))
                row.update(diff_pct=100 * (o.mean() / tick.mean() - 1), p_runs=p_t, p_time=p_dm, coins_better=int((d.mean(axis=0) < 0).sum()),
                           verdict=verdict(o.mean() - tick.mean(), p_t, p_dm))
            rows.append(row)
    lim = pd.DataFrame(rows)
    lim.to_csv(os.path.join(CSV, f"v4_{TAG}_limits.csv"), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 100)
    print(t[["ours", "model", "runs", "test_coin_bars", "mse_mean", "mse_sd", "mse_har", "mse_diff_pct", "mse_diff_lo", "mse_diff_hi", "mse_p_runs", "mse_p_time",
             "mse_coins_better", "mse_verdict", "mse_p_time_holm", "qlike_mean", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_verdict"]].round(4).to_string(index=False))
    print(lim.round(4).to_string(index=False))
    primary = t[(t.ours == HYB_R) & t.model.isin([LGBM, NET_R])].mse_verdict.tolist()
    print("PRIMARY RULE (hybrid with real layers better than LightGBM and than the H-LSTM):", "CONFIRMED" if primary == ["better", "better"] else f"NOT CONFIRMED {primary}")


if __name__ == "__main__":
    {"universe": universe, "download": download, "save": save, "eval": evaluate}[sys.argv[1]]()
