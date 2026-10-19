# Reproducing the paper

The paper uses the pipeline in `src/paper/` and `experiments/paper_*.py`; `src/models/` holds the
network implementations it reuses.

## Environment
- Python 3.12; `uv venv --python 3.12 .venv && uv pip install -r requirements-paper.txt`
  (TA-Lib needs the C library, e.g. `brew install ta-lib`).
- GPU runs were done on NVIDIA L4 GPUs; every run also works on CPU (slower).

## Data
1. `python scripts/download_binance.py` -> `data/binance/` (OHLCV bars) and
   `python scripts/download_binance_1m.py` (one-minute klines for realised variance).
2. `python scripts/download_lunarcrush_paper.py` -> `data/lunarcrush/` (needs a LunarCrush API key in
   the environment variable `LUNARCRUSH_API_KEY` or a `.env` file; the history is revised by the
   vendor, so a new download is a different vintage). LunarCrush data are licensed and are not part
   of this repository.

## Forecasts
`python experiments/paper_run.py configs/paper/final/<config>.yaml [--shard i/n]` writes one `.npz`
per result cell under `results/<name>_<config-hash>/`; runs resume by skipping finished cells.
`configs/paper/final/` holds every configuration of the paper (see the experiment registry, Supplementary Material).

## Tables and figures
All scripts read the saved forecasts and write to `docs/paper/manuscript/tables/` and `figures/`.

| Output files | Script |
|---|---|
| `final_*` (main grids, twins, ablation, budget, fusion, econ, returns) | `experiments/paper_tables_final.py` |
| `datasets.*`, early tables | `experiments/paper_tables.py` |
| `fig_forest`, `fig_ladder`, `fig_equity`, graphical abstract | `experiments/paper_figures_q1.py` |
| `review_*`, `r5_*`, `r8_bench_tests.csv` | `experiments/paper_review_checks.py` |
| `review_audit_*`, `review_bandwidth`, `review_seed_*`, `r5/r6/r7_audit_*` | `experiments/paper_round4_checks.py` |
| `f6_*` | `experiments/paper_fresh6_checks.py` |
| `f7_*` | `experiments/paper_fresh7_checks.py` |
| `dr3_*` (log range as input, group projections) | `experiments/paper_dr3_checks.py` |
| `i7_*` (seeds, level calibration, joint tests, HAR variants) | `experiments/paper_round7_checks.py` |
| `i7c_*` (ten seeds, learning rates, second window) | `experiments/paper_round7_colab.py` |
| `r8_*` (benchmarks, MCS, calibrated economics, exact equivalence, matched twins) | `experiments/paper_round8_checks.py` |
| `r8_audit_formal.csv` | `experiments/paper_audit_formal.py` |
| `r9_*` (Clark-West, trading variants, tuned LightGBM, calendar learning rates) | `experiments/paper_round9_checks.py` |
| `r10_*` (extended LightGBM grid, MCS with tuned LightGBM, social-data equivalence, joint fee test) | `experiments/paper_round10_checks.py` |
| `r11_*` (clipped HAR-XS x frontier, validation selection, final MCS) | `experiments/paper_round11_checks.py` |
| `r6_timing*` | `experiments/paper_timing.py` |

Each script takes function names as arguments, e.g.
`python experiments/paper_round10_checks.py social_tost`; the docstring at the top lists them.
`docs/paper/manuscript/tables/` and `figures/` contain the outputs used in the paper, so every
reported number can be checked without rerunning the models. The saved forecasts (about 5 GB) will
be deposited in a public archive upon publication.
