# Route B evidence consolidation

Zero training. Original projects and prior results are read-only.
P1: frozen-checkpoint margin measurements on own trajectories and shared states.
P2: file-level provenance audit, preserving unknown and failed evidence.
P3: primary-source prior-art audit, including action-gap and PBRS assumptions.

Execution is gated on local inference compatibility, pinned imports, 80 actions,
TTC5 current-state selection, checkpoint hashes, and per-cell smoke validation.
No inference results or causal claims are assumed before the tests complete.

Runtime: the existing isolated environment at
/home/abc/temp/kda-vl-env imports FLA but its Mamba extension requires a newer
glibc than the host provides. The existing Ubuntu22.04 CUDA Docker image provides
glibc2.35. A temporary container runs the same environment without installing or
upgrading anything. /home/abc is mounted read-only; only this audit directory is
mounted writable. The existing mamba_env container is unchanged.

The primary collector is run-p1.py. It refuses to overwrite DONE.json or
ERROR.json, validates the frozen manifest, checks all six cells for every
checkpoint, and stops on any nonzero child exit. The sample is fixed at20/cell.
summarize-case-clusters.py adds descriptive case-cluster intervals only after
primary collection completes. audit-provenance.py and audit-contrasts.py only
read existing results. check-pbrs.py and test-metrics.py are CPU unit checks.
make-report.py renders the final integrated report from completed evidence.

Raw outcomes in P1 are instrumentation context, not a new method comparison.
The 11 trained checkpoints are not11 independent seeds. Shared-state shadow
inference does not execute a new policy or validate navigation improvement.
# Route B revision

The original protocol and frozen manifest remain unchanged. The user amendment is
recorded in protocol-amendment-v2.json. It adds eligible full-score Decision Gap,
separate all-unsafe fallback statistics, speeds, predicted clearance, and selected
actions via deterministic replay of saved commands, without training or policy
search. Original P1 records are preserved. No causal reward claim is authorized.
P3 covers Clipped Advantage Learning and task-aware prediction evaluation as well.
PBRS policy invariance does not rule out finite-training improvement. This audit
does not cancel direction-1 Reward work or authorize new training.

