# MPPI Gain Ablation: Same-Device Pairing

Status: VERIFIED. Evaluation completed with zero network training. No method-level progress claimed.

## Direct Answer

The preregistered H1 verdict is multistep_search. H1 reproduces none of the positive MPPI gain: SR falls by 4.2-13.4pp and CR increases by 7.4-14.6pp across the four cells. Adding an explicit goal-progress objective alone is not an adequate replacement.

Preserving the native score matters: H1V (development-frozen lambda=1.0) improves KDA SR by 4.4pp in square and 3.4pp in circle. However, it reproduces only 49/157 = 31.2% of the pooled MPPI net SR gain (paired-bootstrap 95% CI16.3%-45.1%) and remains below MPPI in every cell. Its GRU square CR increases by 1.4pp [0.2, 2.8]. Progress scoring therefore has a partial benefit, but these two registered single-step alternatives do not explain most of the observed MPPI benefit.

This supports the multistep planning controller as a package, not a unique attribution to search depth or to MPPI sampling. Future-geometry checks, safe-plan admission and warm-start structure have not been individually isolated. The conclusion is conditional on these frozen checkpoints, worlds and registered rules; it does not reject every possible progress-guided one-step controller or establish an independent algorithmic contribution.

## Main Results

| Mother/cell | Base S/C/T | MPPI S/C/T | H1 S/C/T | H1V S/C/T | H1 gain recovery | Preregistered interpretation |
|---|---|---|---|---|---|---|
| gru/square | 403/29/68 | 449/24/27 | 377/66/57 | 414/36/50 | -56.5% | multistep_search |
| gru/circle | 481/10/9 | 491/7/2 | 414/76/10 | 480/11/9 | -670.0% | multistep_search |
| kda/square | 370/37/93 | 438/29/33 | 349/86/65 | 392/38/70 | -30.9% | multistep_search |
| kda/circle | 454/17/29 | 487/13/0 | 391/90/19 | 471/18/11 | -190.9% | multistep_search |

Each cell n=500 paired cases311000-311499. SR/CR/TO percentages equal counts/5.

Primary equal-cell summed SR-gain recovery: -112.7%; classification: multistep_search.
The threshold is unchanged: >=70% progress-information-dominant; <=30% multistep-search-dominant; otherwise mixed.
This operational ablation is not an exclusive causal percentage. MPPI uses future safe-continuation certification and may abort its recovery window; H1 uses current safety only. Its success or failure cannot isolate search depth from future feasibility and exploration direction.

H1V: run with one development-selected global lambda; no test-based lambda selection.

## Paired Effects

| Mother/cell | Comparison | Delta SR pp [95% CI] | Delta TO pp [95% CI] | Delta CR pp [95% CI] | Rescue/damage |
|---|---|---|---|---|---|
| gru/square | mpc_vs_base | 9.2 [6.2, 12.2] | -8.2 [-11.2, -5.4] | -1.0 [-2.2, 0.0] | 55/9 |
| gru/square | h1_vs_base | -5.2 [-9.2, -1.0] | -2.2 [-5.8, 1.4] | 7.4 [4.8, 10.0] | 44/70 |
| gru/square | h1_vs_mpc | -14.4 [-18.0, -11.0] | 6.0 [3.4, 8.8] | 8.4 [6.0, 11.0] | 9/81 |
| gru/square | h1v_vs_base | 2.2 [-0.6, 5.0] | -3.6 [-6.2, -1.0] | 1.4 [0.2, 2.8] | 30/19 |
| gru/square | h1v_vs_mpc | -7.0 [-9.8, -4.4] | 4.6 [2.2, 7.2] | 2.4 [1.2, 3.8] | 8/43 |
| gru/square | h1v_vs_h1 | 7.4 [3.4, 11.2] | -1.4 [-4.8, 2.0] | -6.0 [-8.8, -3.2] | 71/34 |
| gru/circle | mpc_vs_base | 2.0 [0.6, 3.6] | -1.4 [-2.6, -0.2] | -0.6 [-1.8, 0.4] | 13/3 |
| gru/circle | h1_vs_base | -13.4 [-17.0, -9.8] | 0.2 [-1.6, 2.0] | 13.2 [10.0, 16.4] | 14/81 |
| gru/circle | h1_vs_mpc | -15.4 [-18.8, -12.2] | 1.6 [0.4, 3.0] | 13.8 [10.8, 17.0] | 3/80 |
| gru/circle | h1v_vs_base | -0.2 [-2.2, 1.6] | 0.0 [-1.4, 1.4] | 0.2 [-1.2, 1.6] | 11/12 |
| gru/circle | h1v_vs_mpc | -2.2 [-4.0, -0.6] | 1.4 [0.2, 2.8] | 0.8 [-0.2, 2.0] | 4/15 |
| gru/circle | h1v_vs_h1 | 13.2 [9.8, 16.6] | -0.2 [-1.8, 1.4] | -13.0 [-16.2, -10.0] | 78/12 |
| kda/square | mpc_vs_base | 13.6 [9.8, 17.6] | -12.0 [-15.8, -8.4] | -1.6 [-3.0, -0.4] | 88/20 |
| kda/square | h1_vs_base | -4.2 [-9.0, 0.6] | -5.6 [-9.8, -1.6] | 9.8 [6.6, 13.0] | 66/87 |
| kda/square | h1_vs_mpc | -17.8 [-22.0, -13.6] | 6.4 [3.0, 9.8] | 11.4 [8.4, 14.4] | 20/109 |
| kda/square | h1v_vs_base | 4.4 [1.0, 7.8] | -4.6 [-7.8, -1.4] | 0.2 [-1.2, 1.6] | 50/28 |
| kda/square | h1v_vs_mpc | -9.2 [-12.8, -5.6] | 7.4 [4.0, 10.8] | 1.8 [0.4, 3.2] | 23/69 |
| kda/square | h1v_vs_h1 | 8.6 [4.0, 13.4] | 1.0 [-2.8, 4.8] | -9.6 [-12.8, -6.4] | 93/50 |
| kda/circle | mpc_vs_base | 6.6 [4.2, 9.0] | -5.8 [-8.0, -3.8] | -0.8 [-2.2, 0.6] | 37/4 |
| kda/circle | h1_vs_base | -12.6 [-16.6, -8.6] | -2.0 [-4.6, 0.6] | 14.6 [11.2, 18.0] | 27/90 |
| kda/circle | h1_vs_mpc | -19.2 [-22.8, -15.6] | 3.8 [2.2, 5.6] | 15.4 [12.2, 18.8] | 3/99 |
| kda/circle | h1v_vs_base | 3.4 [0.8, 6.0] | -3.6 [-5.8, -1.4] | 0.2 [-1.2, 1.6] | 30/13 |
| kda/circle | h1v_vs_mpc | -3.2 [-5.2, -1.4] | 2.2 [1.0, 3.6] | 1.0 [-0.4, 2.4] | 4/20 |
| kda/circle | h1v_vs_h1 | 16.0 [12.2, 19.8] | -1.6 [-3.8, 0.6] | -14.4 [-17.8, -11.2] | 94/14 |

CI: 10000 paired case bootstrap draws, seed20261017; all cells share the same resampled case indices for pooled quantities. Descriptive intervals are not multiplicity-corrected. Rescue/damage = reference failure -> treatment success / reference success -> treatment failure.

## Inference and Physical Contact

| Mother/cell | Arm | Local inference ms/decision | Swept contacts/500 | Original mean discounted return |
|---|---|---|---|---|
| gru/square | base | 22.122 | 28 | 0.340684 |
| gru/square | mpc | 46.857 | 23 | 0.422559 |
| gru/square | h1 | 22.393 | 62 | 0.295787 |
| gru/square | h1v | 26.775 | 35 | 0.359770 |
| gru/circle | base | 21.613 | 9 | 0.481129 |
| gru/circle | mpc | 44.753 | 7 | 0.495517 |
| gru/circle | h1 | 22.624 | 73 | 0.322768 |
| gru/circle | h1v | 26.226 | 11 | 0.484564 |
| kda/square | base | 24.811 | 36 | 0.299479 |
| kda/square | mpc | 53.835 | 28 | 0.391095 |
| kda/square | h1 | 26.185 | 81 | 0.244342 |
| kda/square | h1v | 27.030 | 37 | 0.329282 |
| kda/circle | base | 24.746 | 16 | 0.428429 |
| kda/circle | mpc | 50.376 | 13 | 0.470140 |
| kda/circle | h1 | 25.938 | 89 | 0.277528 |
| kda/circle | h1v | 26.699 | 17 | 0.451699 |

Inference is synchronized total inference seconds / total executed decisions; base uses the first replay only. All arms use the same local RTX3060, but another agent trains concurrently, so the measured ms describe shared-load operation rather than an isolated throughput benchmark.

## Navigation Efficiency

Each entry is the number of common-success pairs and the treatment-minus-base mean navigation time in seconds. Negative values mean faster navigation within that common-success subset.

| Mother/cell | MPPI n / delta seconds | H1 n / delta seconds | H1V n / delta seconds |
|---|---|---|---|
| gru/square | 394 / -1.900 | 333 / -2.383 | 384 / -1.124 |
| gru/circle | 478 / -0.087 | 400 / +0.451 | 469 / -0.385 |
| kda/square | 350 / -1.103 | 283 / -2.025 | 342 / -0.594 |
| kda/circle | 450 / -0.292 | 364 / -0.042 | 441 / -0.701 |

The subsets differ between comparisons. A faster surviving subset does not compensate for destroyed successes or higher collision rates; it is not an unconditional population efficiency effect.

## Determinism and Cross-Device Differences

| Mother/cell | Exact local replay pairs | Local vs 4090 terminal mismatch count | Mismatching cases (local/4090) |
|---|---|---|---|
| gru/square | 500/500 | 1 | 311277:timeout/success |
| gru/circle | 500/500 | 0 | none |
| kda/square | 500/500 | 1 | 311405:timeout/success |
| kda/circle | 500/500 | 0 | none |

Local replay equality is exact for actions, all robot/human states, real history, all candidate windows and scores, rewards, terminal flags, return, steps, path length and swept clearance. A streaming SHA retains this evidence without saving trajectories. Both aggregates are retained in base and base-repeat.
All efficacy comparisons above use only new local records. Cross-device differences are descriptive and were not a stop criterion. Numerical environment is a competing explanation, not an isolated cause: hardware and library versions were not varied independently.

## Sources and Reproduction

- protocol.json remains byte-identical; AMENDMENT-SAMEDEVICE.md records the user-authorized pairing change. RESULT-round3.md, summary-round3.json and the original failure traceback remain intact.
- local-frozen-manifest.json and parent frozen-manifest.json identify code, mothers, configurations, original reference and original data. Their hashes were verified at start/end.
- local-run.py imports the frozen episode function. MPPI uses the original frozen Controller directly. H1/H1V reuse the original trigger/dwell/fallback and verbatim free_value(); only action search is substituted.
- Units and smoke: local-paired/unit-tests.json and smoke-verified.json. An extra base pair explicitly covers old failing case311277. Every formal base case is again replayed twice.
- Raw primary episodes: local-paired/episodes/test/{gru,kda}/{square,circle}/{base,base-repeat,mpc,h1}/. H1V, if required, uses its frozen numeric lambda in the directory name.
- The bootstrap pairs numeric case IDs, not individual control steps or independent training seeds. Reused test worlds and two fixed mothers do not establish population-level generalization or novelty.
- No network training, reward change, new safety rule, forbidden case or remote compute.
- H1 is a simple monotonic goal-progress baseline. This is a mechanism ablation, not a new algorithm.

## Reproduction Commands

Private Python: /workspace/kda-history-replication-20261010/position-only-20261015/runtime/bin/python inside mamba_env.
local-run.py units -> local-run.py smoke --workers 2 -> local-run.py full --workers 2 -> local-analyze.py h1.
If H1V is required: local-run.py development -> local-analyze.py freeze-lambda -> local-run.py h1v-smoke -> local-run.py h1v -> local-analyze.py final.
All scripts are under /home/abc/workspace/mppi-ablation-20261017/.

## H1V Development Selection

One global lambda = 1.0, selected only from development cases310000-310199 by the frozen success/collision/timeout/small-lambda tie break.

| Lambda | Development S/C/T (n=800) |
|---|---|
| 0.1 | 683/33/84 |
| 0.3 | 696/30/74 |
| 1.0 | 708/36/56 |
| 3.0 | 692/61/47 |

## Final Raw-Data Validation

- All 16 formal mother/scene/arm counts were independently recounted from raw episodes; all rescue/damage counts agree with their case-paired records.
- All 2000 base replay pairs match exactly for states/history/windows/scores trace, action and selection SHA, terminal outcome, time, return, path and swept clearance.
- 582 no-action-change negative controls match the local base; mismatches: 0.
- The selected lambda was frozen before every H1V test file was created. Source/weight/config hashes and the original protocol hash remain unchanged.
- Actual formal evaluation used 3 single-thread workers; the reproduction examples with 2 workers are valid commands, not the recorded concurrency. Exact executed commands and timing are in local-paired/pipeline-state.json and final-validation.json.
- Same-device comparisons to the preserved previous local partial batch: local-paired/cross-local-partial-check.json (273 cases, no differences). These old records are not used for effect estimation.

## Secondary H1V Gain Capture

These fractions are descriptive. They do not replace the frozen H1 70/30 decision rule.

| Mother/cell | H1V gain / local MPPI gain | Conditional paired-bootstrap 95% CI |
|---|---|---|
| gru/square | 23.9% | -7.1% to 48.8% |
| gru/circle | -10.0% | -250.0% to 66.7% |
| kda/square | 32.4% | 8.7% to 53.8% |
| kda/circle | 51.5% | 16.7% to 77.4% |

Equal-cell pooled secondary H1V capture: 31.2% [16.3%, 45.1%].
Ratio intervals condition on a positive bootstrapped MPPI gain; positive-denominator fractions are recorded in summary.json. In particular, the small GRU circle gain makes its ratio imprecise.

H1/H1V recompute the action at each active step; neither holds one command constant for the entire 16-step recovery window.
The comparison concerns an explicit progress scoring rule, not newly observed goal information. It does not isolate search depth, future-safe-plan certification, proposal/warm-start structure or optimizer choice from one another.
The previous cross-device terminal divergence was a real success/timeout difference, not a floating-point clearance-assertion false positive. Its precise cause remains unisolated.
This frozen-policy ablation does not establish a cause in MC supervision, KDA memory or reward-shaping training, and establishes no new independent algorithmic contribution.

## Completion and Tooling Record

The formal pipeline wall time was 3h11m18s, excluding the initial smoke/setup. The round4 run contains 8000 formal comparison episodes, 2000 exact base repeats, 3200 development episodes and 102 smoke/extra-regression executions: 13302 total. All evaluation, selection, analysis and final source-verification commands exited0; no evaluation workers remain running. The other agent's training and the previously paused job were not modified.

The postprocessing syntax check initially failed to write a root-owned bytecode-cache directory from the host account. It was repaired by compiling in memory under the actual private Python with bytecode writing disabled; retry exited0. No permissions, experiment logic, data or results were changed. Details: local-paired/postprocessing-notes.json. Final raw validation exited0: local-paired/final-validation.json.

The current artifact integrity record is final-local-integrity.json. The unchanged final-integrity.json is the historical round3 snapshot; its old report/summary hashes match the preserved RESULT-round3.md and summary-round3.json copies, not the replacement current report/summary.
