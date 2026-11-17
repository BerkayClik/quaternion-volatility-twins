# A hierarchical quaternion LSTM for multi-source cryptocurrency volatility forecasting, tested against real-valued twins

Code for a twin-controlled test of a hierarchical quaternion LSTM (one quaternion encoder per data
source, meta-quaternion fusion, calendar term) for next-bar volatility (log range) forecasting on 15
cryptocurrencies at hourly, 4-hourly and daily frequency, with Binance market data and LunarCrush
social and sentiment data. The model is compared with HAR, seasonal HAR, GARCH and pooled LightGBM
benchmarks and with real-valued twins of the same depth and a matched number of parameters, for
recurrent, attention and convolutional encoders, at several depths, parameter budgets and learning
rates.

The study has two stages on the same data, folds and training protocol. Stage 1 (`configs/paper/final/`)
holds the benchmarks, the one-layer recurrent and attention pairs and their controls. Stage 2
(`configs/paper/v2/`) holds the proposed two-layer model, the depth and convolution grid, the calendar
term, the fusion and source controls, and the seed, budget and learning-rate controls.

## Contents

| Path | What |
|---|---|
| `src/paper/` | features, groups, walk-forward folds, model zoo, training protocol, baselines, evaluation, economic evaluation |
| `src/models/` | quaternion LSTM (step-wise and fused), quaternion Transformer, quaternion temporal CNN, hierarchical encoders and fusion operators, real twins |
| `experiments/paper_run.py` | resumable runner; one `.npz` per result cell |
| `experiments/paper_v2_*.py` | scripts that compute every table and figure of the paper from the saved forecasts |
| `experiments/paper_*.py` | scripts of stage 1 (benchmarks with week cells, lagged LightGBM-S, one-layer controls) |
| `configs/paper/v2/`, `configs/paper/final/` | every experiment configuration of stage 2 and stage 1 |
| `scripts/` | data download (Binance, LunarCrush) |
| `docs/paper/manuscript_v2/tables`, `figures` | the tables (CSV and LaTeX) and figures of the paper |
| `docs/paper/manuscript/tables`, `figures` | tables and figures of stage 1 |
| `tests/` | exactness test of the fused quaternion LSTM; tests of the quaternion convolution, the depth suffix and the matched twins |

See [REPRODUCE.md](REPRODUCE.md) for the environment, data download, runs and the script behind
each table.

## Data

Binance data are public and are downloaded by the scripts. LunarCrush social and sentiment data
are licensed and cannot be redistributed; the download script re-obtains them with a LunarCrush
subscription. No data are included in this repository.

## License

MIT (see `LICENSE`).
