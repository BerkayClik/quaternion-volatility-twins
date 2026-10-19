# Quaternion and real hierarchical encoders for multi-source cryptocurrency volatility forecasting

Code for a twin-controlled comparison of quaternion and real-valued neural encoders (flat and
hierarchical LSTMs and Transformers) for next-bar volatility (log range) forecasting on 15
cryptocurrencies at hourly, 4-hourly and daily frequency, with Binance market data and LunarCrush
social and sentiment data, against HAR, seasonal HAR, GARCH and pooled LightGBM benchmarks.
Every quaternion model has a real twin with matched structure and a near-equal parameter budget.

## Contents

| Path | What |
|---|---|
| `src/paper/` | features, groups, walk-forward folds, model zoo, training protocol, baselines, evaluation, economic evaluation |
| `src/models/` | quaternion LSTM (step-wise and fused), quaternion Transformer, hierarchical encoders and fusion operators, real twins |
| `experiments/paper_run.py` | resumable runner; one `.npz` per result cell |
| `experiments/paper_*.py` | scripts that compute every table and figure from the saved forecasts |
| `configs/paper/final/` | every experiment configuration |
| `scripts/` | data download (Binance, LunarCrush) |
| `docs/paper/manuscript/tables`, `figures` | the resulting tables (CSV and LaTeX) and figures |
| `tests/` | exactness test of the fused quaternion LSTM |

See [REPRODUCE.md](REPRODUCE.md) for the environment, data download, runs and the script behind
each table.

## Data

Binance data are public and are downloaded by the scripts. LunarCrush social and sentiment data
are licensed and cannot be redistributed; the download script re-obtains them with a LunarCrush
subscription. No data are included in this repository.

## License

MIT (see `LICENSE`).
