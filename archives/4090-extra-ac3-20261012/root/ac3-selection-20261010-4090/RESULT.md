# AC3 Frozen-Weight Development Test

**Status: REJECT; 6000 episodes; zero training.**

## Question
Does action-coverage selection improve navigation beyond TTC5 and a one-direction Goal-CV1s selector?

## Frozen Setup
KDA T24 fresh IL50+RL10000 paired seeds419/443. 20-person circle/square,500cases/cell,90000-90499developmentblock. TTC5/Goal-CV1s/AC3, same80smoothedcommands, alpha0.3, unchanged reward/filter/value network. All20humans remain in reward/filter; only5networkslots change. No92000confirmationcases.

The two downloaded files are the method sources; latest user protocol supersedes their384-episode sample plan. Canonical TTC robot float32roundtrip/tie behavior is preserved in AC3fallback/fill.

## Results
Source: per-case/s{seed}-{rule}/cell{4,5}.json. Each overall row is1000episodes; eachcell500.

| Seed | Selector | Cell | SR % | CR % | TO % | Mean Original Return | Success Time s |
|---|---|---|---:|---:|---:|---:|---:|
| 419 | ttc5 | overall | 83.30 | 7.40 | 9.30 | 0.361050 | 18.138 |
| 419 | ttc5 | 4 | 88.80 | 7.00 | 4.20 | 0.406853 | 17.523 |
| 419 | ttc5 | 5 | 77.80 | 7.80 | 14.40 | 0.315248 | 18.841 |
| 419 | goal-cv1s | overall | 79.90 | 10.70 | 9.40 | 0.337809 | 17.617 |
| 419 | goal-cv1s | 4 | 83.20 | 11.20 | 5.60 | 0.358567 | 17.609 |
| 419 | goal-cv1s | 5 | 76.60 | 10.20 | 13.20 | 0.317051 | 17.625 |
| 419 | ac3 | overall | 84.40 | 7.30 | 8.30 | 0.369574 | 17.905 |
| 419 | ac3 | 4 | 89.00 | 6.20 | 4.80 | 0.411860 | 17.489 |
| 419 | ac3 | 5 | 79.80 | 8.40 | 11.80 | 0.327289 | 18.369 |
| 443 | ttc5 | overall | 81.80 | 5.10 | 13.10 | 0.355978 | 18.606 |
| 443 | ttc5 | 4 | 91.20 | 2.40 | 6.40 | 0.436418 | 17.759 |
| 443 | ttc5 | 5 | 72.40 | 7.80 | 19.80 | 0.275538 | 19.674 |
| 443 | goal-cv1s | overall | 83.40 | 5.20 | 11.40 | 0.362304 | 18.779 |
| 443 | goal-cv1s | 4 | 88.80 | 5.00 | 6.20 | 0.405204 | 18.145 |
| 443 | goal-cv1s | 5 | 78.00 | 5.40 | 16.60 | 0.319405 | 19.500 |
| 443 | ac3 | overall | 82.00 | 4.20 | 13.80 | 0.367498 | 18.163 |
| 443 | ac3 | 4 | 92.60 | 2.20 | 5.20 | 0.452530 | 17.360 |
| 443 | ac3 | 5 | 71.40 | 6.20 | 22.40 | 0.282467 | 19.204 |

## Paired Comparisons
Positive Return/SR favors the first selector; positive CR/TO/NavTime is worse. 95%caseIDclusterbootstrap intervals (10000draws), not160k independentsteps or a trainingseedpopulation test. Pooled intervals keep bothseeds andbothcells together for eachcase.

| Seeds | Contrast | Return Delta [95% CI] | SR Delta pp | CR Delta pp | TO Delta pp | Rescue / Damage | Common Success Time Delta s |
|---|---|---|---:|---:|---:|---|---:|
| 419 | ac3-ttc5 | 0.008524 [-0.011476, 0.028722] | 1.10 | -0.10 | -1.00 | 92 / 81 | -0.211 |
| 419 | ac3-goal-cv1s | 0.031765 [0.007247, 0.056139] | 4.50 | -3.40 | -1.10 | 137 / 92 | -0.035 |
| 419 | goal-cv1s-ttc5 | -0.023241 [-0.046690, 0.000276] | -3.40 | 3.30 | 0.10 | 92 / 126 | -0.513 |
| 443 | ac3-ttc5 | 0.011520 [-0.005831, 0.029238] | 0.20 | -0.90 | 0.70 | 88 / 86 | -0.307 |
| 443 | ac3-goal-cv1s | 0.005194 [-0.014682, 0.025303] | -1.40 | -1.00 | 2.40 | 106 / 120 | -0.477 |
| 443 | goal-cv1s-ttc5 | 0.006326 [-0.014039, 0.026893] | 1.60 | 0.10 | -1.70 | 112 / 96 | 0.148 |
| 419+443 | ac3-ttc5 | 0.010022 [-0.003109, 0.023390] | 0.65 | -0.50 | -0.15 | 180 / 167 | -0.259 |
| 419+443 | ac3-goal-cv1s | 0.018479 [0.003640, 0.033601] | 1.55 | -2.20 | 0.65 | 243 / 212 | -0.257 |
| 419+443 | goal-cv1s-ttc5 | -0.008457 [-0.023552, 0.006871] | -0.90 | 1.70 | -0.80 | 204 / 222 | -0.179 |

## Selection Diagnostics
Source: per-case/*/cell*.steps.jsonl.gz; actual constraint means minimum actual next-position endpointclearance, not a swept collisioncertificate. Rates are on each method own trajectories and cannot be compared as same-state causal interventions.

| Seed / Rule | Steps | Selection Different % | TTC Omits Actual Constraint % | Selected Adds Omitted Constraint % | All Unsafe % |
|---|---:|---:|---:|---:|---:|
| s419-ttc5 | 80887 | 0.00 | 26.36 | 0.00 | 2.03 |
| s419-goal-cv1s | 77750 | 82.62 | 25.76 | 25.59 | 2.37 |
| s419-ac3 | 78820 | 19.90 | 26.58 | 12.90 | 2.25 |
| s443-ttc5 | 88276 | 0.00 | 24.97 | 0.00 | 1.63 |
| s443-goal-cv1s | 86755 | 82.90 | 24.42 | 24.33 | 1.45 |
| s443-ac3 | 88334 | 16.58 | 24.78 | 10.61 | 1.35 |

## Decision
Frozen gate: positive Returnvsbothcontrols ineachseed, nonnegativeSR and nonincreasingcollisionsvsboth. These strict point-estimate guards are a conservative operational prototype gate, not proof of noninferiority. Full CIs andpaircounts are in summary.json.

```json
[
  {
    "seed": 419,
    "baseline": "ttc5",
    "return_positive": true,
    "SR_nonnegative": true,
    "CR_nonincrease": true
  },
  {
    "seed": 419,
    "baseline": "goal-cv1s",
    "return_positive": true,
    "SR_nonnegative": true,
    "CR_nonincrease": true
  },
  {
    "seed": 443,
    "baseline": "ttc5",
    "return_positive": true,
    "SR_nonnegative": true,
    "CR_nonincrease": true
  },
  {
    "seed": 443,
    "baseline": "goal-cv1s",
    "return_positive": true,
    "SR_nonnegative": false,
    "CR_nonincrease": true
  }
]
```

## Evidence Boundaries
- Verified: correctness/smoke checks, pairedclosed-loop outcomes and allinputSHAverification.
- Notverified: novelty, generalization beyondthese2weights, independentconfirmations, causal benefitofanyindividual補選step.
- Existing23.48%/11737stepomission claim is userprovidedcontext, not independently reproduced here.
- 90000block is developmentdata alreadyused before, notafreshholdout.
- Runtime isremote4090existingtorch2.9.1 stack; all3arms inthiscomparison areevaluatedhere. No cross-machinebitwiseparity isclaimed.
- SmallerCV1sgeometryerror andbetterselectionhit do not prove betterlong-runvalue.

## Recommendation
Stop AC3. No horizon/weight/threshold tuning, no retraining, no additional seeds or92000confirmations to rescue it. The measured failure does not invalidate all action-aware representations.

No new experiment is started automatically.
