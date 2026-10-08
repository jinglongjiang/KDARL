# KDARL: KDA-VL Research Code and Evidence

This repository publishes the current scene-level KDA-VL project, its Mamba and
GRU controls, completed experiment evidence, and the completed human-selection
study with independent confirmation. It does **not** publish model weights, ORCA training datasets,
the parent repository's Git history, or the discontinued `shixu` research line.

## Architecture

```text
ORCA imitation initialization (Monte Carlo value regression)
  -> relational spatial encoder
  -> 24-frame scene embeddings, width 256
  -> 4-layer Mamba / GRU / official FLA KDA temporal backbone
  -> scalar history-conditioned value
  -> 80-action one-step value lookahead
  -> inherited safety/risk processing and action execution
  -> online Monte Carlo value refinement
```

The KDA implementation is the thin adapter in
`CrowdNav/crowd_nav/policy/kda_temporal.py`, using FLA's
`KimiDeltaAttention`. This is not an entire Kimi Linear language model, an
actor-memory predictor, PPO, or a newly invented KDA operator.

## Three Experimental Versions

| Version | Training procedure | Source | Results |
|---|---|---|---|
| V1 | IL50 + 3,000 online MC episodes | `versions/v1/CrowdNav` (parent commit `b216efd`) | `CrowdNav/artifacts/vl-3000-screening` |
| V2 | Resume V1 for another 7,000 episodes; 10,000 cumulative; teacher replay rebuilt | `versions/v2/CrowdNav` (parent commit `1964f05`) | `CrowdNav/artifacts/vl-10000-continuation` |
| V3 | Reuse IL, train 10,000 RL episodes continuously; vectorized scorer and MC-only replay storage | `CrowdNav` (parent commit `86969cd`) | `CrowdNav/artifacts/vl-optimized-20261007` |

All versions train only on five-person circle environments. These are
experimental versions, not three different KDA architectures. Three frozen
backbone arms are preserved within each version.

### Existing 192-Episode Results

Six environments: 5/10/20 people, circle/square, 32 cases per cell; training
seed 419. Rates below are recomputed from stored episode counts.

| Version | Backbone | Success | Collision | Timeout |
|---|---|---:|---:|---:|
| V1 | Mamba | 88.54% | 6.77% | 4.69% |
| V1 | GRU | 86.98% | 7.29% | 5.73% |
| V1 | KDA | 83.33% | 10.94% | 5.73% |
| V2 | Mamba | 85.94% | 7.81% | 6.25% |
| V2 | GRU | 85.94% | 9.90% | 4.17% |
| V2 | KDA | 88.02% | 7.81% | 4.17% |
| V3 | Mamba | 86.98% | 10.42% | 2.60% |
| V3 | GRU | 82.81% | 8.85% | 8.33% |
| V3 | KDA | 84.90% | 9.90% | 5.21% |

V1/V2 use cases 88000-88031; V3 uses 89000-89031. Cross-version changes
do not isolate code optimization, and these single-seed, small-sample results
do not establish stable KDA superiority. V2 is not uninterrupted training:
online replay/RNG/environment counters were not fully restored. Negative
results and failed preflight attempts remain in the evidence directories.

## Repository Map

| Directory / file | Contents |
|---|---|
| `CrowdNav/crowd_nav` | Training, state contracts, temporal/value models, replay and inherited policy dependencies |
| `CrowdNav/crowd_sim` | Original simulator and ORCA/robot/human execution dependencies |
| `CrowdNav/tools` | Fixed-budget training, continuation, optimization and correctness tools |
| `versions/v1`, `versions/v2` | Historical source snapshots, independent of current optimized source |
| `CrowdNav/artifacts/kda-vl-freeze` | Reference/causal/save-load/shared-init checks and non-weight trace fixtures |
| `CrowdNav/artifacts/vl-*` | Protocols, final evaluations, comparisons, completion metadata, training logs and plots |
| `experiments/diagnostics-20261008` | Completed D0-D9 diagnostic scripts, JSON results and compressed decision traces |
| `experiments/selection-20261008` | Completed first5/nearest5/existing-TTC Top-5 study: 60 cell JSON files, 30,000 episode executions, frozen protocol and independent confirmation |
| `snapshot-manifest.json` | SHA256, sizes and provenance of exported files; excluded binary inventory; study state at export |

Compressed `*.json.gz` files contain decision-level JSON evidence, not weights.
Compressed `trajectory.log.gz` files retain the original rollout logs.
Historical absolute paths and checkpoint hashes are retained for provenance;
they do not imply that those files are available after cloning.

### Reports

- `CrowdNav/KDA_VL_DROPIN_FREEZE.md`: parent reconstruction and module correctness.
- `CrowdNav/VL-v2-contract.md`: common state/action/value contracts.
- `CrowdNav/KDA-VL-3000-SCREENING.md`: initial fixed-budget experiment.
- `CrowdNav/KDA-VL-COMPREHENSIVE-REPORT.md`: V1 and V2 training/evaluation report.
- `CrowdNav/KDA-DIAGNOSTIC-REPORT.md`: completed high-resolution diagnostic audit.
- `CrowdNav/KDA-SELECTION-CONFIRMATION-REPORT.md`: verified development and independent confirmation results, paired tests, safety regressions and evidence limits (Chinese).

The comprehensive report predates V3: do not mistake its V2 results for the
latest continuous-run weights. The diagnostic audit subsequently evaluated
V3 weights on 3,000 episodes per arm; that larger evaluation is distinct from
the 192-episode comparison above.

## Completed Selection Study

The previous commit `0674f41` was an incomplete selection-study snapshot. This
update includes both completed blocks, the frozen winner, final verdict and
report verification. No weights, mixed-density training or retraining are added.

- Development: cases 90000-90499; first5 / nearest5 / inherited TTC Top-5.
- Winner frozen from KDA development results: `ttc5`.
- Independent confirmation: cases 91000-91499; first5 versus frozen `ttc5`.
- Verdict: `CONFIRMED_SELECTION_GAIN` under the predeclared KDA adoption rule.

| KDA independent confirmation, 3,000 episodes | First5 | TTC Top-5 | Change |
|---|---:|---:|---:|
| Success | 84.83% | 90.27% | +5.43 pp |
| Collision | 12.07% | 7.17% | -4.90 pp |
| Timeout | 3.10% | 2.57% | -0.53 pp (not significant after correction) |

These are deployment-selection gains with one frozen five-person-trained seed,
not evidence of a KDA-specific memory advantage or superiority over Mamba.
GRU also benefits, but its timeout rate rises from 8.00% to 9.63%. Local KDA
regressions and original-success damage are retained in the report. KDA ran on
RTX4090 and GRU on RTX3060 with different PyTorch versions: cross-arm absolute
performance comparisons are not permitted in this study. Six cells reuse case
IDs; the report includes case-cluster bootstrap intervals in addition to the
requested exact paired McNemar tests and Bonferroni correction.

Use `snapshot-manifest.json` for exact export time, file hashes and captured statuses.

## Dependencies and Running

The actual formal training environment was RTX4090, Python 3.10, PyTorch 2.9.1,
and `mamba_ssm` 2.2.6.post3, with FLA supplied through `PYTHONPATH`. The initial
freeze environment is separately recorded in
`CrowdNav/artifacts/kda-vl-freeze/environment-manifest.json`; it is not the same
runtime as later formal training. Consult each protocol rather than combining
the two environments.

Official dependency provenance recorded by the freeze:

- FLA: https://github.com/fla-org/flash-linear-attention,
  commit `9f38d24980c46d46bd38614e743cdacd21906578`.
- Kimi Linear algorithm source: https://github.com/MoonshotAI/Kimi-Linear,
  commit `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`.
- Mamba: the installed official `mamba_ssm` package; see environment manifests.
- Simulator: `rvo2`, Gym, NumPy and Matplotlib; training uses PyTorch.

The repository intentionally keeps imported parent registry dependencies, some
of which are optional and inactive in KDA-VL. They are not additional tested
algorithms or new KDA components. An unused, invalid standalone memory snippet
is not exported; the active replay implementation is `utils/ppo_buffer.py`.

For evaluation, provide your own matching checkpoint and run from the selected
project root. Always initialize the action grid from the loaded config and
assert **80** actions. For example:

```bash
python experiments/selection-20261008/evaluate-selection.py --help
python experiments/diagnostics-20261008/eval6.py --help
cd CrowdNav
python tools/test_vl_training_contract.py
```

`run-study.py` and the archived overnight controllers retain machine-specific
paths and SSH destinations. Review/configure these before use; they are not
portable unattended launchers. No authentication material is supplied.

No formal training or new navigation evaluation was started for this export.
The inherited simulator/project license is preserved in `LICENSE`.
