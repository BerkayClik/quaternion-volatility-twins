# Analysis plans of the paper and when they were fixed

Every analysis listed here had its plan (the data, the models, the comparison and, where one exists, the criterion of
success) committed to the development repository of the project before it was run or computed, except where a row says
otherwise. The plan is the docstring of the named script or
the named configuration at the given commit; the dates are commit times (UTC+3). Results were always reported as
they came out; where a result went against the expectation the text says so (for example the two-by-two of model class
and input representation, after which the paper was retitled).

The commit identifiers are those of the development repository. The public code release is a snapshot of that
repository in a single commit, without its history, so the commits cannot be looked up in the release. The docstrings and configurations in the release are the plans as they were
committed, with two kinds of later change: additions (a further plan appended to a file, which has its own row
below, and usage lines), and seven places (six docstrings and one configuration header) where a request quoted in
another language was replaced by an English paraphrase.

| Commit | Date | Analysis | Where the plan is |
|---|---|---|---|
| `9f49d56` | 2026-11-17 18:50 | The two candidate designs of the network, each with a real twin of equal size, before any run | configs under `configs/paper/v3/` |
| `980536c` | 2026-11-17 20:21 | One further fold on the bars after the sample (the 33 days) | `experiments/paper_v3_confirm.py`, configs under `configs/paper/v3/` |
| `66f6e05` | 2026-11-17 23:05 | Design controls of the network (inputs, calendar term, fusion, depth, single encoder, budgets, learning rates, one model per coin) | configs under `configs/paper/v3/` |
| `424ff0a` | 2026-11-17 23:15 | Recent benchmark networks (DLinear, PatchTST, iTransformer) on the inputs of the network | configs under `configs/paper/v3/` |
| `65b8106` | 2026-11-18 00:16 | Weighted average, stacking and cascading of the network and LightGBM | `experiments/paper_v3_hybrid.py` |
| `11a6cb5` | 2026-11-18 00:38 | Cascade: network trained on the residual of LightGBM | `experiments/paper_v3_cascade.py`, `configs/paper/v3/a_cascade_pool_4h.yaml` |
| `adddc15` | 2026-11-18 02:37 | The two versions of the network at 500, 1,000 and 2,000 parameters, ten runs | configs under `configs/paper/v3/` |
| `49d292f` | 2026-11-18 09:40 | Hybrids and cascades of the network and HAR (the two plain averages with HAR had been computed a few minutes before this plan and are post hoc; the fitted schemes and the cascades had not) | `experiments/paper_v3_hybrid_har.py`, `experiments/paper_v3_cascade.py` |
| `e8c815c` | 2026-11-18 10:51 | Five runs of every model (mean and standard deviation of MSE and QLIKE) | `experiments/paper_v3_runs.py` |
| `9943319` | 2026-11-18 23:14 | Three-model hybrids with HAR (averages, stacking, cascades) | `experiments/paper_v3_triple.py`, configs `configs/paper/v3/a_cascade_har_*`, `a_oof_har_*` |
| `28b39f7` | 2026-11-20 10:16 | Effect of the six data types | `experiments/paper_v3_inputs.py` |
| `ca0586b` | 2026-11-20 18:29 | The two tests (Welch over runs, Diebold-Mariano over time) and the rule | `experiments/paper_v3_tests.py` |
| `f3e4322` | 2026-11-22 19:41 | Hourly and daily bars | `experiments/paper_v3_bars.py` |
| `247f048` | 2026-11-22 20:49 | The forecasts as 95% range limits | `experiments/paper_v3_paper.py` (`limits`) |
| `71a3f38` | 2026-11-23 18:46 | Tests of the version with real layers against the single models | `experiments/paper_v4_paper.py` |
| `2ef1bc1` | 2026-11-23 21:29 | Held-out coins: selection rule, models and primary criterion, before any of their data were downloaded | `experiments/paper_v4_newcoins.py` |
| `84fb1f7` | 2026-11-23 21:31 | Averages with a third model, selection on the validation block, coverage tests, encompassing, realised variance, controls of the real version | `experiments/paper_v4_round2.py` |
| `2edeaa1` | 2026-11-23 21:34 | Further benchmarks (GARCH, GJR-GARCH, a second boosting library) and tuning of the benchmarks | `experiments/paper_v4_bench.py`, `configs/paper/v4/tune_*` |
| `1c0464c` | 2026-11-24 01:47 | Two-by-two of model class and input representation, with the reading fixed in advance; held-out earlier period for both sets of coins | `experiments/paper_v4_round3.py` (points 1 and 2) |
| `d41affc` | 2026-11-24 02:33 | Averages of equal cost, the hybrid on the variance scale, limits at 99% and 99.5% with a quantile model, pooled linear benchmarks and CARR | `experiments/paper_v4_round3.py` (point 4) |
| `db18b85` | 2026-11-24 03:27 | Averages of several runs on the four data sets, the earlier period without the validation overlap, test bandwidths, the extreme bar, the hybrid with a single LSTM, spread of the two-by-two gains | `experiments/paper_v4_round4.py` |
| `f1ae53a` | 2026-11-27 17:38 | The quaternion hybrid as the proposed system: every analysis from `71a3f38` on recomputed with the quaternion hybrid in the place of the hybrid with the real twin (one routine with the version as a parameter, required to reproduce the earlier tables first), with the reading of each result fixed before the numbers; the network and the hybrid against their real twins on six data sets | `experiments/paper_v5_stats.py` |
| `44c0bc3`, `241dd6f` | 2026-11-28 13:26, 13:27 | One fold on the bars after the sample for both sets of coins (point 3 of `1c0464c`): 44 days from 14 October to 28 November 2026, run once on 2026-11-28 instead of the day before submission, reported for the quaternion hybrid; models, evaluation and reading fixed before the download of the new bars and before any fit. The second commit moves the end of the block one bar earlier, to the last bar that was complete at the download, before anything was downloaded | `experiments/paper_v5_post.py`, `configs/paper/v4/post_*` |
| `e20efd7` | 2026-11-28 17:54 | Five checks after an automated external review of the manuscript: paired test over runs; QLIKE under other level constants and without the extreme bar; the quantile fit of LightGBM with a constant, coverage tests and the finite-sample quantile in the tail; ranking of the new coins on volumes before the earlier period; the real twin with a projection and with the initialisation of the quaternion layers | `experiments/paper_v5_round5.py`, `configs/paper/v5/twinctl_pool_4h.yaml` |


Analyses without a plan fixed in advance (descriptive; listed in the Limitations of the paper): the model confidence
sets, the statistics of the figure on forecast combination, the fitted weights of the three-model schemes, and the
comparison of the hybrid with its network on the 33 days after the sample.

Which version is presented as the system. The plans from `71a3f38` to `db18b85` call the hybrid with the real twin
"the hybrid" and name it in their criteria; the quaternion hybrid is a secondary comparison there. The paper presents
the quaternion hybrid, the system of the plans before `71a3f38` and the one with which the development was carried
out. This choice was made on 2026-11-27 with all results known; it is not the outcome of a test, and the paper says
so (sections "Design history" and "Held-out data"). The plan of `f1ae53a` was written with every earlier result known:
it fixes how each statement is read for the quaternion hybrid, not what is tested. The comparison of the two versions
on the six data sets (the table on the real twin) is descriptive.

Deviations from a plan:
- The configurations of `9f49d56` say that the choice between the two candidate designs of the network is made on the
  validation blocks. The design that the paper takes is the one with the lower error on the four test blocks and on
  the fold after the sample, as the section "Design history" of the paper states; the validation errors were not the
  stated basis of the choice.
- No comparison or criterion of a later plan was changed after its results were seen.

The earlier design steps (the earlier quaternion designs and the two candidate designs of 2026-11-17) are described in
the section "Design history" of the paper.
