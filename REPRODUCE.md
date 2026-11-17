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
`python experiments/paper_run.py --config configs/paper/v2/<config>.yaml --out results/<folder> [--shard i/n]`
(or a configuration in `configs/paper/final/`) writes one `.npz` per result cell under
`results/<folder>/<name>_<config-hash>/`; runs resume by skipping finished cells.
`configs/paper/v2/` and `configs/paper/final/` hold every configuration; each file states in its header what it tests and when it was fixed.

## Where the analysis scripts look for the runs
The scripts of the paper read the runs from the folders in which they were produced. To reproduce the
tables, write each configuration to the folder given here (`--out results/<folder>`).

| Folder under `results/` | Configurations |
|---|---|
| `v2/out` | `v2/depth_pool_4h`, `v2/depth_80000_pool_4h` |
| `v2ctl/out` | `v2/cnn_lr3e4_pool_4h`, `v2/depth_single_4h`, `v2/depth_pool_1d` |
| `v2cal/out` | `v2/cal_pool_4h` |
| `v2f/out` | `v2/cal_pool_1d`, `v2/cal_pool_1h` |
| `v2g/out` | `v2/fusion_pool_4h`, `v2/sources_pool_4h` |
| `v2h/out` | `v2/seeds_pool_4h` |
| `v2i/out` | `v2/lr3e3_cal_pool_4h`, `v2/lr3e3_sources_pool_4h`, `v2/tfmatch_cal_pool_4h`, `v2/seeds_sources_pool_4h`, `v2/seeds_hcnn_pool_4h` |
| `v2j/out` | `v2/budget_2000_cal_pool_4h`, `v2/budget_5000_cal_pool_4h`, `v2/budget_80000_cal_pool_4h` |
| `v2k/out` | `v2/tfmatch_pool_4h` |
| `v2local` | `v2/lgbm_seeds_4h`, `v2/lgbm_seeds_1h`, `v2/lgbm_seeds_1d`; `v2_benchx_4h` is written by `paper_v2_round2.py benchx_save` |
| `final_fa/out` | `final/single_4h`, `final/pool_4h`, `final/single_1d`, `final/pool_1d`, `final/return_4h`, `final/ablation_pool_4h` |
| `final_fb/out` | `final/single_1h` |
| `final_fc/out` | `final/calendar_pool_4h` |
| `final_local` | `final/har_season_*`, `final/harx_*`, `final/lgbm_*`, `final/lgbm_seeds_*` |
| `final_tm/out`, `final_cr/out`, `final_dr2/out` | `final/twinmatch_pool_4h`; `final/return_pool_4h`; `final/grouproj_pool_4h` |
| `final_fz/out`, `final_r7a/out`, `final_rc/out` | `final/budget_*_pool_4h`; `final/lr3e3_pool_4h`, `final/lr3e4_pool_4h`; `final/randgroups_pool_4h` |
| `r8_local`, `r11_local` | HAR with week cells and the lagged LightGBM-S (`paper_round8_checks.py harsx_save lgbm_save`), HAR-XS with week cells (`paper_round11_checks.py harxsx_save`) |

## Tables and figures of the paper (stage 2)
The scripts read the saved forecasts in `results/` and write to `docs/paper/manuscript_v2/tables/` and
`figures/`. The forecasts of stage 2 are produced by the configurations in `configs/paper/v2/`; the
benchmarks and one-layer controls come from stage 1 (`configs/paper/final/` and the stage-1 scripts
below, which save HAR with week cells, the lagged LightGBM-S and the tuned benchmarks in the runner's
format).

| Output | Command |
|---|---|
| `v2_main` (main table: MSE, QLIKE, 95% limits), `v2_robust` (three bar lengths) | `python experiments/paper_v2_round2.py main` / `robust` |
| `v2_twins_all.csv` (all 75 twin tests, one Holm family), `fig_twins.pdf` | `python experiments/paper_v2_round2.py twins`, then `python experiments/paper_v2_figures.py twins` |
| `v2_depth` (errors by encoder family and depth) | `python experiments/paper_v2_round2.py depth` (after `twins`) |
| `v2_scope.csv`, `fig_scope.pdf` (parameter budget, learning rate) | `python experiments/paper_v2_round2.py scope`, then `python experiments/paper_v2_figures.py scope` |
| `v2_depth_models.csv`, `v2_depth_twins.csv` | `python experiments/paper_v2_depth.py` |
| `v2_seeds10.csv` (ten seeds, intervals over seeds and time) | `python experiments/paper_v2_round2.py seeds` |
| `v2_ablation`, `v2_fusion`, `v2_sources` | `python experiments/paper_v2_tables.py` |
| `v2_bench`, `v2_benchx`, `v2_equal_width`, `v2_small_checks`, `v2_qlike_bar`, `v2_exceed`, `v2_blocks`, `v2_selection`, `v2_levels`, `v2_returns_pooled` | `python experiments/paper_v2_round2.py <name>` (see the docstring; `benchx_save` fits the linear benchmark variants first) |
| `v2_seed_sensitivity`, `v2_single_seed`, `v2_sources_v1`, `v2_returns` | `python experiments/paper_v2_checks.py all` |
| `fig_data.pdf` | `python experiments/paper_v2_figures.py data` |

`python experiments/paper_v2_round2.py all` runs every comparison. The naming of the models in the
configurations: `-L<n>` sets the depth (`lstm-L2`, `hq_meta-L2`, `qcnn-L6`), `+cal` adds the calendar
term, `cnn`/`qcnn` and `hc_<fusion>`/`hqc_<fusion>` are the temporal CNN twins, and the configuration key
`match: {real: quaternion}` sizes a real twin to the parameter count of its quaternion twin.

## Tables and figures of stage 1
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
`docs/paper/manuscript_v2/tables/` and `figures/` (and `docs/paper/manuscript/` for stage 1) contain the
outputs used in the paper, so every reported number can be checked without rerunning the models. The saved forecasts (about 5 GB) will
be deposited in a public archive upon publication.
