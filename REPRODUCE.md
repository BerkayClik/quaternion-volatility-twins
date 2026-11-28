# Reproducing the paper

The paper uses the pipeline in `src/paper/` and the scripts `experiments/paper_*.py`; `src/models/` holds the
network implementations. Sections 1 to 5 concern the paper; section 6 concerns the earlier design stages, whose
benchmark forecasts the paper reuses.

## 1. Environment
- Python 3.12; `uv venv --python 3.12 .venv && uv pip install -r requirements-paper.txt`
  (TA-Lib needs the C library, e.g. `brew install ta-lib`).
- GPU runs were done on NVIDIA L4 GPUs; every run also works on CPU (slower). On macOS run with `--threads 1`.
- `python -m pytest tests -q` runs the unit tests.

## 2. Data
1. `python scripts/download_binance.py` -> `data/binance/` (OHLCV bars of the 15 coins of the study) and
   `python scripts/download_binance_1m.py` (one-minute klines, for the check against realised variance).
2. `python scripts/download_lunarcrush_paper.py` -> `data/lunarcrush/` (needs a LunarCrush API key in the
   environment variable `LUNARCRUSH_API_KEY` or a `.env` file; the vendor revises its history, so a new download
   is a different vintage). LunarCrush data are licensed and are not part of this repository.
3. The 15 held-out coins: `python experiments/paper_v4_newcoins.py universe` applies the selection rule of the
   paper to the exchange's listings and volumes (the outcome used in the paper is `configs/paper/v4/newcoins.json`),
   and `python experiments/paper_v4_newcoins.py download` writes their bars to `data/binance/`.

The samples of the 15 coins of the study are the rows on which both sources exist (Section 3.1 of the paper), so
rebuilding the development data exactly needs the vendor files, although the proposed system reads exchange data
only. The three held-out data sets are built from the exchange files alone (`exchange_only: true` in their
configurations) and need no vendor data.

## 3. Forecasts
`python experiments/paper_run.py --config configs/paper/<stage>/<config>.yaml --out results/<folder> [--shard i/n]`
writes one `.npz` per result cell (test and validation forecasts) under `results/<folder>/<name>_<config-hash>/`;
runs resume by skipping finished cells. Each configuration states in its header what it tests and when it was
fixed. Benchmarks that are not networks are fitted and saved in the same format by the commands named below.

### Where the scripts of the paper look for the runs
The scripts find the runs of `configs/paper/v3/a_*` under any `results/v3*/out/` and the tuning runs under any
`results/v4tune*/out/`; the other folders are fixed. The folders used for the paper:

| Folder under `results/` | Written by |
|---|---|
| `v3local` | configurations `v3/gattn_range_pool_4h`, `v3/meta_exchange_pool_4h` (the two candidate designs of the network) |
| `v3a/out` ... `v3q/out` | configurations `v3/a_*` (design controls, budgets, learning rates, five runs, bar lengths, data types, cascades, out-of-fold forecasts) |
| `v3confirm/out`, `v3confirm/bench` | configurations `v3/confirm_*` on the bars after the sample (`data_post/`, see the docstring of `paper_v3_confirm.py`); `paper_v3_confirm.py save` |
| `v5post/out`, `v5post/bench_study`, `v5post/bench_new` | configurations `v4/post_*` on the 44 days after the sample for both sets of coins (a new download in `data_sub/`, see the docstring of `paper_v5_post.py`); `paper_v5_post.py save` |
| `v5twin/out` | configuration `v5/twinctl_pool_4h`: the real twin with a projection per group, and with the initialisation of the quaternion layers as well (check 5 of `paper_v5_round5.py`) |
| `v3runs` | `paper_v3_runs.py lgbm`, `paper_v3_bars.py lgbm` (runs 3 and 4 of LightGBM), `paper_v3_inputs.py lgbm` (LightGBM on the other input sets), `paper_v3_triple.py third` |
| `v4bench/garch`, `hgb`, `lgbm_grid` | `paper_v4_bench.py garch`, `hgb`, `lgbm_grid` |
| `v4tune/out`, `v4tuneb/out` | configurations `v4/tune_*` (learning rates of the networks; instance normalisation for PatchTST and iTransformer) |
| `v4new/out`, `v4new/bench` | configuration `v4/new_pool_4h` (held-out coins); `paper_v4_newcoins.py save` |
| `v4early/out`, `v4early/bench_study`, `bench_new` | configurations `v4/early_study_pool_4h`, `v4/early_new_pool_4h` (earlier period); `paper_v4_round3.py early_save study`, `early_save new` |
| `v4views/out`, `v4views/lgbmw`, `v4views/linear` | configuration `v4/views_smlp_pool_4h` (network on the summaries); `paper_v4_round3.py lgbm_window` (trees on the window); `paper_v4_round3.py fit_extra` (quantile LightGBM, pooled linear models, CARR) |
| `r8_local`, `final_local`, `final_fa/out`, `final_fb/out`, `final_fc/out`, `v2local`, `v2cal/out`, `v2f/out` | benchmarks and networks of the earlier stages that the paper reuses (section 6): HAR, the seasonal HAR and runs 0 to 2 of the lagged LightGBM-S are saved by `paper_round8_checks.py harsx_save lgbm_save` |

## 4. Tables and figures of the paper
All scripts read the saved forecasts in `results/` and write result tables (CSV) to
`docs/paper/manuscript_v2/tables/`; `paper_v5_paper.py tables figs` then writes the LaTeX tables and the
figures of the paper to `docs/paper/manuscript_v5/`. No model is trained by these commands unless the table says
so. Each script lists in its docstring what it computes, what it reads and what was known when its plan was
fixed (`docs/paper/PLANS.md` gives the order). The tables `v4_*` hold both hybrids (column `ours` or `version`);
the paper uses the rows of `H-QLSTM + LightGBM`. `paper_v5_stats.py check` runs the routine of the `v5_*` tables
with the real twin and requires the numbers of the corresponding `v4_*` tables; `paper_v5_stats.py all` writes
the `v5_*` tables.

| In the paper | Result table (CSV) | Command (`python experiments/...`) |
|---|---|---|
| Table 2, main results (`main.tex`) | `v5_main` | `paper_v5_stats.py main` |
| Table 3, held-out data (`confirm.tex`) | `v4_newcoins`, `v4_early_study`, `v4_early_new` (and `*_limits`), `v5_sets_averages` | `paper_v4_newcoins.py eval`; `paper_v4_round3.py early study`, `early new`; `paper_v5_stats.py sets` |
| Table 4, 95% range limits (`limits.tex`) | `v4_limits`, `v4_r2_coverage` | `paper_v4_paper.py limits`; `paper_v4_round2.py coverage` |
| Table 5, limits at three levels (`levels.tex`) | `v5_levels` | `paper_v4_round3.py fit_extra` (fits the quantile model), then `paper_v5_stats.py extra` |
| Table 6, combination schemes (`combos.tex`) | `v4_combos`, `v5_averages` | `paper_v4_paper.py stats`; `paper_v5_stats.py averages` |
| Table 7, model class and input representation (`views.tex`) | `v5_views`, `v5_gains` | `paper_v4_round3.py lgbm_window` (fits the trees on the window), then `paper_v5_stats.py views` |
| Table 8, pairs of models (`partners.tex`) | `v5_partners` | `paper_v5_stats.py partners` |
| Table 9, the real twin (`twin.tex`) | `v5_twin` | `paper_v5_stats.py twin` |
| Table 10, data types (`inputs.tex`) | `v3_inputs`, `v5_inputs` | `paper_v3_inputs.py`; `paper_v5_stats.py inputs` |
| Table 11, tuned benchmarks (`tuned.tex`) | `v5_tuned`, `v4_tuned_settings`, `v4_tuned_choices` | `paper_v4_bench.py lgbm_grid` (fits the grid), then `eval`; `paper_v5_stats.py tuned` |
| Table 12, bar lengths (`robust.tex`) | `v4_bars` | `paper_v4_paper.py bars` |
| Figure 1, the system | | `docs/paper/manuscript_v5/figures/src/fig_system.tex` (TikZ) |
| Figure 2, data and folds | | `paper_v3_paper.py figs` (`fig_data.pdf`) |
| Figure 3, main results | `v3_paper_models`, `v3_paper_coins`, `v4_tests` | `paper_v3_paper.py stats`; `paper_v4_paper.py stats` |
| Figure 4, range limits | `v4_limits`, `v5_limits_example` | `paper_v4_paper.py limits`; `paper_v5_stats.py example` |
| Figure 5, forecast combination | `v3_paper_corr`, `v3_paper_weight`, `v3_paper_weights`, `v5_time` | `paper_v3_paper.py stats`; `paper_v5_stats.py partners` |
| Figure 6, design controls | `v3_grid`, `v3_grid_seeds`, `v3_grid_bars`, `v3_small` | `paper_v3_tables.py` |

Numbers that appear in the text only:

| Result table (CSV) | What | Command |
|---|---|---|
| `v3_runs`, `v3_tests` | five runs of every model; the two tests and the rule | `paper_v3_runs.py`, `paper_v3_tests.py` |
| `v3_hybrid`, `v3_hybrid_har`, `v3_cascade`, `v3_triple` (and `*_weights`), `v5_triple_weights` | weighted averages, stacking and cascades of two and three models | `paper_v3_hybrid.py`, `paper_v3_hybrid_har.py`, `paper_v3_cascade.py eval`, `paper_v3_triple.py`, `paper_v5_stats.py weights` |
| `v4_bench` | GARCH, GJR-GARCH, a second boosting library | `paper_v4_bench.py garch`, `hgb`, then `eval` |
| `v5_encompass`, `v5_rv`, `v5_regimes`, `v5_selected` | encompassing tests, realised variance as the target, volatility regimes, members selected on validation data | `paper_v5_stats.py encompass`, `rv`, `regimes`, `averages` |
| `v5_cost`, `v5_variance`, `v5_linear` | averages of equal cost, the hybrid on the variance scale, pooled linear benchmarks and CARR | `paper_v5_stats.py extra` |
| `v5_sets_overlap`, `v5_sets_bandwidth`, `v5_sets_print`, `v5_flat` | earlier period without the validation overlap, test bandwidths, the extreme bar of 10 October 2025, the hybrid with a single quaternion LSTM | `paper_v5_stats.py sets`, `flat` |
| `v4_r2_controls` | the design controls with real layers | `paper_v4_round2.py controls` |
| `v3_confirm`, `v3_test_blocks`, `v4_confirm` | the 33 days after the sample (descriptive) | `paper_v3_confirm.py eval`, `test`; `paper_v4_paper.py extras` |
| `v4_post_study`, `v4_post_new` (and `*_limits`), `v5_post_twin` | the 44 days after the sample (14 October to 28 November 2026), both sets of coins, plan fixed before the download | `paper_v5_post.py eval`, `twin` |
| `v5_r5_paired`, `v5_r5_qlike`, `v5_r5_limits`, `v5_r5_volume`, `v5_r5_twinctl` | five checks after an external review of the manuscript: the paired test over runs, QLIKE under other level constants and without the extreme bar, the range limits in the tail and the quantile fit with a constant, the ranking of the new coins on earlier volumes, the real twin matched in projection and initialisation; plan fixed before they were run | `paper_v5_round5.py paired`, `qlike`, `limits`, `volume`, `twinctl` |

Every loss ratio in these files is the loss averaged over the 15 coins and all test bars of one model divided by
that of the other, from the means of the five runs. The result tables and figures used in the paper are part of
this repository, so every reported number can be checked without rerunning the models. The saved forecasts that
the scripts read are attached to the release `forecasts-v5` of this repository (five archives, about 1.3 GB; unpack
them in the root of the repository).

## 5. Names of the models in the configurations
`hl_gattn` and `hq_gattn` are the network of the hybrid with real and with quaternion layers (one encoder per
feature group, attention over the group states); `-L<n>` sets the depth (`lstm-L2`, `qcnn-L6`); `+cal` adds the
calendar term to the earlier designs; `cnn`/`qcnn` and `hc_<fusion>`/`hqc_<fusion>` are the temporal CNNs;
`smlp` is the feed-forward network on the lagged summaries; the key `match: {real: quaternion}` sizes a real
network to the parameter count of its quaternion counterpart; `exchange_only: true` builds the sample from the
exchange file alone.

## 6. Earlier design stages
Stage 2 (`configs/paper/v2/`) and stage 1 (`configs/paper/final/`) are the earlier designs of the network
(section "Design history" of the paper). The scripts read the runs from the folders in which they were produced:

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
| `v2s/out` | `v2/seeds_budget_2000_cal_pool_4h_twin` |
| `v2w/out` | `v2/sources_width_pool_4h` |
| `v2local` | `v2/lgbm_seeds_4h`, `v2/lgbm_seeds_1h`, `v2/lgbm_seeds_1d`, `v2/carr_4h`, `v2/seeds_budget_2000_cal_pool_4h`; `v2_benchx_4h` and `v2_lgbmlags16_4h` are written by `paper_v2_round2.py benchx_save` and `lgbmlag16_save` |
| `final_fa/out` | `final/single_4h`, `final/pool_4h`, `final/single_1d`, `final/pool_1d`, `final/return_4h`, `final/ablation_pool_4h` |
| `final_fb/out` | `final/single_1h` |
| `final_fc/out` | `final/calendar_pool_4h` |
| `final_local` | `final/har_season_*`, `final/harx_*`, `final/lgbm_*`, `final/lgbm_seeds_*` |
| `final_tm/out`, `final_cr/out`, `final_dr2/out` | `final/twinmatch_pool_4h`; `final/return_pool_4h`; `final/grouproj_pool_4h`, `final/rangein_pool_4h` |
| `final_fz/out`, `final_r7a/out`, `final_rc/out` | `final/budget_*_pool_4h`; `final/lr3e3_pool_4h`, `final/lr3e4_pool_4h`; `final/randgroups_pool_4h` |
| `r8_local`, `r11_local` | HAR with week cells and the lagged LightGBM-S (`paper_round8_checks.py harsx_save lgbm_save`), HAR-XS with week cells (`paper_round11_checks.py harxsx_save`) |

### Stage 2
The scripts write to `docs/paper/manuscript_v2/tables/` (`v2_*`) and `figures/`.

| Output | Command |
|---|---|
| `v2_main`, `v2_robust` (three bar lengths) | `python experiments/paper_v2_round2.py main` / `robust` |
| `v2_twins_all.csv` (75 tests of quaternion against real networks of equal size), `fig_twins.pdf` | `python experiments/paper_v2_round2.py twins`, then `python experiments/paper_v2_figures.py twins` |
| `v2_depth` (errors by encoder family and depth) | `python experiments/paper_v2_round2.py depth` (after `twins`) |
| `v2_scope.csv`, `fig_scope.pdf` (parameter budget, learning rate) | `python experiments/paper_v2_round2.py scope`, then `python experiments/paper_v2_figures.py scope` |
| `v2_seeds10.csv` (ten seeds at 20,000 and at 2,000 parameters) | `python experiments/paper_v2_round2.py seeds` |
| `v2_ablation`, `v2_fusion`, `v2_sources` | `python experiments/paper_v2_tables.py` |
| `v2_bench`, `v2_benchx`, `v2_equal_width`, `v2_small_checks`, `v2_qlike_bar`, `v2_exceed`, `v2_blocks`, `v2_selection`, `v2_levels`, `v2_returns_pooled`, `v2_mcs`, `v2_lrselect`, `v2_rangein` | `python experiments/paper_v2_round2.py <name>` (see the docstring; `benchx_save` fits the linear benchmark variants first) |
| `v2_seed_sensitivity`, `v2_single_seed`, `v2_sources_v1`, `v2_returns`, `v2_vintages` (two downloads of the vendor data) | `python experiments/paper_v2_checks.py all` |
| `fig_data.pdf` | `python experiments/paper_v2_figures.py data` |

### Stage 1
The scripts write to `docs/paper/manuscript/tables/` and `figures/`.

| Output files | Script |
|---|---|
| `final_*` (main grids, quaternion against real networks, ablation, budget, fusion, econ, returns) | `experiments/paper_tables_final.py` |
| `datasets.*`, early tables | `experiments/paper_tables.py` |
| `fig_forest`, `fig_ladder`, `fig_equity` | `experiments/paper_figures_q1.py` |
| `review_*`, `r5_*`, `r8_bench_tests.csv` | `experiments/paper_review_checks.py` |
| `review_audit_*`, `review_bandwidth`, `review_seed_*`, `r5/r6/r7_audit_*` | `experiments/paper_round4_checks.py` |
| `f6_*` | `experiments/paper_fresh6_checks.py` |
| `f7_*` | `experiments/paper_fresh7_checks.py` |
| `dr3_*` (log range as input, group projections) | `experiments/paper_dr3_checks.py` |
| `i7_*` (seeds, level calibration, joint tests, HAR variants) | `experiments/paper_round7_checks.py` |
| `i7c_*` (ten seeds, learning rates, second window) | `experiments/paper_round7_colab.py` |
| `r8_*` (benchmarks, model confidence set, calibrated economics, exact equivalence, matched networks) | `experiments/paper_round8_checks.py` |
| `r8_audit_formal.csv` | `experiments/paper_audit_formal.py` |
| `r9_*` (Clark-West, trading variants, tuned LightGBM, calendar learning rates) | `experiments/paper_round9_checks.py` |
| `r10_*` (extended LightGBM grid, social-data equivalence, joint fee test) | `experiments/paper_round10_checks.py` |
| `r11_*` (clipped HAR-XS x frontier, validation selection) | `experiments/paper_round11_checks.py` |
| `r6_timing*` | `experiments/paper_timing.py` |

Each script takes function names as arguments, e.g.
`python experiments/paper_round10_checks.py social_tost`; the docstring at the top lists them.
