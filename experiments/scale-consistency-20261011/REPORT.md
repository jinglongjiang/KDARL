# Scale Consistency Audit

Status: FAILED_CLOSED at the revised saved-field parity gate. New 12/15-person evaluation episodes: 0. Training episodes: 0.

## Revised Gate Result

CONFIRMED FAILURE: seed443 KDA-T24, 20-person square, first mismatching case 91004.
Cases 91000-91003 in this worker passed all revised saved-field conditions.

| Field | Historical | New 3060 run | Absolute difference | Gate |
|---|---|---|---:|---|
| outcome | timeout | timeout | 0 | equal |
| time | 50.25 s | 50.25 s | 0 | equal |
| selection trace SHA256 | e075016c6d7ccf745e7240b4067c8694500d0f32d2fa9ce2ea6a91cd418a2ccd | 53bf7129883d1244d9e9ca903d37d80a444351993be8d3409cb9e0e822c36279 | n/a | FAILED |
| return | -0.06910088094291078 | -0.0692263330590076 | 0.00012545211609682472 | FAILED; tolerance 0.0001 |
| minimum clearance | 0.192986231258371 m | 0.19457353328146898 m | 0.0015873020230979806 m | FAILED; tolerance 0.0001 m |

Verified runtime: new RTX3060, Python3.11.17, torch2.7.1+cu126. Historical seed443 files record torch2.9.1+cu128; training/evaluation provenance identifies the original4090 stack. This includes GRU: do not substitute provenance from the earlier GRU-selection experiment.

Verified fail-closed execution: the mismatching worker exited1; the other two active workers were terminated with code-15; the pending fourth parity worker did not start. Container exited1 without automatic restart or OOM. New-cell smoke and full16000 evaluations never started. All935 frozen inputs, including eight weights, passed end verification.

First divergent step: UNVERIFIED and unavailable. Historical per-step trajectories do not exist. The first mismatching CASE is91004, not a claimed first divergent STEP. Any saved action trace is from this new run only and does not reconstruct the historical trajectory.

Source: parity/20-person/square_crossing/s443-kda-t24.json.mismatch.json, corresponding partial.json, job-evidence/, status.json and closeout.json.

## Phenomenon

| People | Geometry | A disagreement | Independent expectation | Ratio |
|---|---|---:|---:|---:|
| 5 | circle | 1.25% | 1.73% | 0.7214 |
| 5 | square | 3.10% | 3.63% | 0.8546 |
| 10 | circle | 3.60% | 4.59% | 0.7850 |
| 10 | square | 7.40% | 8.55% | 0.8654 |
| 20 | circle | 15.60% | 17.84% | 0.8746 |
| 20 | square | 28.75% | 37.68% | 0.7629 |

These are descriptive means across four overlapping A comparisons.

| People | Geometry | A: change seed | B: change backbone | C: change T |
|---|---|---:|---:|---:|
| 5 | circle | 1.25% | 1.35% | 1.35% |
| 5 | square | 3.10% | 3.20% | 3.20% |
| 10 | circle | 3.60% | 3.70% | 3.60% |
| 10 | square | 7.40% | 7.50% | 7.90% |
| 20 | circle | 15.60% | 15.60% | 15.40% |
| 20 | square | 28.75% | 31.75% | 29.65% |

VERIFIED: B/C are of similar descriptive magnitude to A on these fixed weights.
This does not order architecture/T causal effects versus the population variance of training.

## Question And Evidence

- VERIFIED: all 48 old cell files contain 500 matched cases each; eight weight hashes match file metadata.
- VERIFIED: all old files contain outcome, return, clearance, time and selection-trace hash, but no argmax trajectory.
- VERIFIED: the user replaced bitwise action-trajectory parity with a saved-field equivalence gate; that revised gate failed as detailed above.
- VERIFIED: effective old time limit is 50 seconds; some old episodes run beyond 25 seconds. Do not change to 25.
- UNVERIFIED: 12/15-person behavior; no new environment episodes executed.

## Options And Recommendation

Stop at the failed revised gate, as instructed. No new cells or automatic runtime switches.
The two-stage4090-replay proposal was withdrawn at user request and was NOT executed.
Next requires an explicit decision about evaluating in a matching numerical stack or revising comparability requirements; do not relax tolerances or mix old/new results silently.
Software and hardware stacks differ, but the specific cause of the mismatch is UNPROVEN. This is not evidence of a KDA architecture defect or a new scale-consistency result.

## Interpretation Limits

A ratio below one measures positive paired success association, not proof of independence.
A flat ratio alone does not identify ceiling effects or explain generalization.
B/C disagreement at least as large as A cannot imply B/C is no larger than A; report magnitudes without reversing the inequality.
C is interpreted as changing T at fixed seed/backbone; the supplied list explicitly defines only A and B.
All confidence intervals concern sampled cases conditional on the eight fixed weights, not training-seed population variance.
Count and density remain confounded. No backbone superiority, stability or novel algorithm claim.

Engineering note: [eval_protocol] time_limit=25 is a dead configuration key in this environment/harness. [env] time_limit=50 is effective and was asserted. Activating the dead key in a later code change would silently break historical protocol comparability. A50-second limit leaves nominal travel-time slack but does not by itself prove a physical stall.

Sources: consistency-existing.json, provenance-existing.json, frozen-manifest.json.
