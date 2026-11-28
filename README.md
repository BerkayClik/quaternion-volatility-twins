# A hybrid of a hierarchical quaternion LSTM and LightGBM for intraday cryptocurrency volatility forecasting: held-out evaluation and a real-valued twin

Code, experiment configurations and result tables of the paper.

The proposed system is the plain average of two forecasts of the log range of the next four-hour bar: a
hierarchical quaternion LSTM network on a one-week window of bars (H-QLSTM, 19,875 parameters) and LightGBM on
lagged summaries of the same bars (the lagged LightGBM-S, 35 regressors). It is developed on 15 cryptocurrencies with
four expanding walk-forward folds, five training runs per model and two tests (over runs and over time). The
frozen system is then evaluated on three held-out data sets: 15 other coins in the same period, and an earlier
18-month period for both sets of coins. Further analyses look for the source of the gain: blending, stacking
and cascading; a two-by-two of model class and input representation; averages of several runs at equal cost; a
third member; tuned benchmarks; and the forecasts used as limits on the range of the next bar. A real twin of
the network of equal size (H-LSTM, 20,142 parameters) is trained next to it throughout and tests what the
quaternion algebra contributes: the hybrid built with the twin is not less accurate.

## Contents

| Path | What |
|---|---|
| `src/paper/` | features and feature groups, walk-forward folds, model zoo, training protocol, benchmarks (HAR, seasonal HAR, GARCH, CARR, LightGBM), evaluation |
| `src/models/` | the networks: hierarchical encoders and fusion operators, LSTM, quaternion LSTM, Transformer, temporal CNN, and DLinear, PatchTST and iTransformer (`modern_baselines.py`) |
| `experiments/paper_run.py` | resumable runner; one `.npz` file of forecasts per result cell |
| `experiments/paper_v3_*.py`, `paper_v4_*.py`, `paper_v5_*.py` | the scripts that compute every table and figure of the paper from the saved forecasts; each states its plan in its docstring |
| `configs/paper/v3/`, `configs/paper/v4/` | the experiment configurations of the paper |
| `docs/paper/PLANS.md` | the analysis plans and when they were fixed |
| `docs/paper/manuscript_v5/tables`, `figures` | the tables (LaTeX) and figures of the paper |
| `docs/paper/manuscript_v2/tables/v3_*.csv`, `v4_*.csv`, `v5_*.csv` | the result tables behind them |
| `scripts/` | data download (Binance, LunarCrush) |
| `tests/` | unit tests of the pipeline |

See [REPRODUCE.md](REPRODUCE.md) for the environment, the data download, the runs and the command behind each
table and figure.

## Names used in the code

| In the paper | In the code |
|---|---|
| H-QLSTM, the network of the hybrid | `hq_gattn` on the groups price (with the range), market and calendar; "candidate A" in the configurations of `configs/paper/v3/` |
| H-LSTM, its real twin | `hl_gattn` on the same groups; "H-LSTM (real twin)" in some result tables |
| LightGBM | `lgbmsl-pool`, "LightGBM-S, lagged" in the result tables; 35 regressors, one model for all coins |
| the hybrid | plain average of `hq_gattn` and the lagged LightGBM-S, run k with run k ("H-QLSTM + LightGBM" in the result tables; "H-LSTM + LightGBM" is the hybrid with the real twin) |
| seasonal HAR | "HAR-S x": HAR with one dummy per cell of the week |
| development data | the 15 coins of the study on the four test blocks |
| held-out data | `newcoins` (15 other coins, same period), `early_study` and `early_new` (both sets of coins, 30 October 2023 to 22 April 2025) |
| trees on the window, network on the summaries | `lgbmw-pool`, `smlp` |

The docstrings of the scripts were written as plans before the analyses were run and are kept as plans
(`docs/paper/PLANS.md` says what was added later). Two things in them differ from the paper. The scripts
`paper_v4_*.py` call the hybrid with the real twin "the hybrid", because an earlier draft presented that version;
`experiments/paper_v5_stats.py` computes the same analyses for the quaternion hybrid of the paper, and
`docs/paper/PLANS.md` explains the order. And some docstrings use the working name "dual-view hybrid"; the name
left the paper after the two-by-two of model class and input representation
(`experiments/paper_v4_round3.py`, point 1) showed that the two inputs are not the source of the gain.

## Earlier design stages

The network was developed in stages before the hybrid (section "Design history" of the paper). Their code is
part of this repository because the scripts of the paper reuse it: the benchmark forecasts (HAR, seasonal HAR,
the lagged LightGBM-S) and several helper functions come from it.

| Path | What |
|---|---|
| `configs/paper/final/`, `experiments/paper_tables*.py`, `paper_review_checks.py`, `paper_round*_checks.py`, `paper_fresh*_checks.py` and related scripts | stage 1: the benchmarks, and one-layer quaternion and real networks (recurrent and attention) with their controls |
| `configs/paper/v2/`, `experiments/paper_v2_*.py` | stage 2: a two-layer hierarchical quaternion LSTM with a calendar term, and depth, convolution, fusion, source, seed, budget and learning-rate controls |
| `docs/paper/manuscript/tables`, `figures`; `docs/paper/manuscript_v2/tables/v2_*`, `figures` | their result tables and figures |
| `docs/paper/manuscript_v3/tables`, `figures`; `docs/paper/manuscript_v4/tables`, `figures` | tables and figures of two earlier drafts (v3: the quaternion hybrid before the held-out evaluation; v4: the hybrid with the real twin as the default) |

## Data

Binance data are public and are downloaded by the scripts. LunarCrush social and sentiment data, which only the
study of data types uses, are licensed and cannot be redistributed; the download script re-obtains them with a
LunarCrush subscription. No data are included in this repository. The saved forecasts of every model run that the
scripts of the paper read (about 1.3 GB in five archives) are attached to the release `forecasts-v5` of this
repository: unpacked in its root they create `results/`, from which the scripts recompute every table and figure.
A snapshot with a persistent identifier will be deposited in a public archive upon publication.

## License

MIT (see `LICENSE`).
