# B Round-One Scheduling

No command here trains. Run models only under the main Agent's GPU scheduler.
All conditions/cases are frozen in b-contract.json. References need CPU only;
run each reference once, not once per checkpoint (identical physics/config).

```bash
export PYBIN=/root/miniconda3/envs/mamba/bin/python
export PYTHONPATH=/root/kda-vl-deps/flash-linear-attention
export B=/root/ab-causal-isolation-20261011/b
export C=/root/ttc5-causal-20261011
export ACCEPT=/root/ab-causal-isolation-20261011/acceptance.json
```

Ensure b/project contains both public fixes. Cherry-pick SHA differs locally;
acceptance checks accepted policy file SHA, not erroneous ancestor equivalence.
Do not pass the original unmodified project as --project.

## CPU Qualification

Already completed locally: 160 scenes; 32 paired E3 checks x 201 steps, all exact.
Remote qualification reproduces without GPU and without checkpoints:

```bash
CUDA_VISIBLE_DEVICES= "$PYBIN" "$B/run-b.py" cpu --phase smoke --project "$B/project" --config "$C/inputs/kda-s419-t24/config.ini"
CUDA_VISIBLE_DEVICES= "$PYBIN" "$B/run-b.py" cpu --phase main --project "$B/project" --config "$C/inputs/kda-s419-t24/config.ini"
```

## Frozen Batches

The following commands PRINT JSON argv; they never launch jobs. Main scheduler
executes each job's argv exactly once. Run GPU jobs after A extraction, with main
choosing concurrency from the measured smoke latency. CPU references can be
scheduled separately; set CUDA_VISIBLE_DEVICES empty for resource=CPU jobs.

```bash
"$PYBIN" "$B/run-b.py" jobs --phase smoke --project "$B/project" --causal "$C" --acceptance "$ACCEPT"
```

60 jobs x 3 episodes = 180 smoke episodes: 120 learned, 60 reference. A single
job example, useful for first timing (it is already in the printed batch):

```bash
"$PYBIN" "$B/run-b.py" baseline --phase smoke --project "$B/project" --causal "$C" --acceptance "$ACCEPT" --model kda-s419-t24 --controller model --geometry circle --condition E1
```

After all smoke jobs pass, gate before any formal model/reference episodes:

```bash
"$PYBIN" "$B/run-b.py" smoke-gate --project "$B/project"
"$PYBIN" "$B/run-b.py" jobs --phase main --project "$B/project" --causal "$C" --acceptance "$ACCEPT"
```

60 jobs x 16 episodes = 960 formal episodes: 640 learned, 320 reference.
Order case90000..90015, no outcome-dependent sample changes. Circle and square
stay separate. Smoke repeats first3; it is never added to formal denominators.
No 92000 input is read. Overwriting an outcome file is refused.

## B5 Frozen Continuations

After ALL 10 main condition/geometry shards for one model complete, freeze roots
before that model's branch outcomes. At most2 failed cases and2 matched success
controls per condition/geometry. No controls is explicitly not protected evidence.

```bash
for M in kda-s419-t24 kda-s443-t24 gru-s419-t24 gru-s443-t24; do
  "$PYBIN" "$B/run-b.py" roots --project "$B/project" --model "$M"
done
```

Schedule these commands under main GPU control after the model batches:

```bash
for M in kda-s419-t24 kda-s443-t24 gru-s419-t24 gru-s443-t24; do
  for G in circle square; do
    for E in E1 E2 E3 E4 E5; do
      "$PYBIN" "$B/run-b.py" counterfactual --project "$B/project" --causal "$C" --acceptance "$ACCEPT" --model "$M" --geometry "$G" --condition "$E"
    done
  done
done
"$PYBIN" "$B/run-b.py" summarize --project "$B/project"
```

This loop is serial by default; concurrency is owned by main. A0 full physical,
score and history replay must be exact, excluding timing. A1 is prescribed before
continuations, not hindsight-best. Zero-root shards still produce valid empty
branch artifacts. The summary marks missing shards INCOMPLETE.

## Time Accounting

Local CPU E3 smoke: 3.79s for6x201steps, formal about20s for32x201steps. GPU
estimates require first actual smoke; do not assume wall time from FLOPs.
Total model formal <=128640 simulator steps; if measured end-to-end per-step is
50ms, serial upper bound107min plus load/warmup; at100ms214min. References are
independent CPU jobs. Branch cap <=160 roots, <=320 episodes before duplicates.
The runner stops new work 10min before the8h deadline, preserving partial outputs.
No sample shrink/expansion is implicit in a deadline stop.

## Limits

B4 full2x2 training is NOT authorized. B5's newly paired conditions cover5/20;
the old10human-square audit may be read, but cannot silently fill a matched10human
causal arm. State this limit rather than infer high-density amplification from
unmatched old data. E3 proves isolation only for verified trajectories/worlds,
not all possible actions or all20human tasks. CV-MPC/quantized ORCA are known
planning references adapted to this command/filter contract, not optimal bounds.
