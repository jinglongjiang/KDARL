"""Record an explicit amendment without modifying the original frozen protocol."""
import json
from pathlib import Path
from common import BASE, save, sha, verify

out = BASE / 'protocol-amendment-v2.json'
assert not out.exists()
save(out, dict(
    version='route-b-v2', original_protocol_sha256=sha(BASE/'protocol.json'),
    original_frozen_manifest_sha256=sha(BASE/'frozen-manifest.json'),
    frozen_inputs_verified=verify(),
    timing='User amendment after original own-policy collection completed; not retrospective preregistration.',
    preserved_own_cells=len(list((BASE/'p1-margin/per-checkpoint').glob('*/cell[0-5].json'))),
    paused_controller=json.loads((BASE/'status.json').read_text()),
    order=['P3 revised prior audit','P2 completed provenance audit','P1 supplementary smoke/cost gate','P1 supplementary full measurement'],
    measurements=dict(
        value_gap='Top two successor V, no reward/risk/filter; not Q action gap.',
        raw_gap='Top two r + gamma V before safety/risk/tie penalty.',
        decision_gap='Top two complete scores among clearance>=0.2 candidates if n_safe>=2; n_safe==1 gives null.',
        all_unsafe='Separate fallback top-two complete-score gap among all 80; never pooled into safe decision gap.',
        complete_score='raw - 0.8*max(0,0.2-predicted_clearance) - 0.001*(index==0)',
        additional=['top5 V speeds and speed spread','top5 predicted clearance','final selected index and speed','top1/top2 velocity difference','actual one-step geometry with explicit robot-invisibility condition']),
    recovery='Deterministically replay saved executed commands, verify original geometry/outcomes/selection. No new policy rollout or training; common-state metrics reuse enriched reference.',
    cost_gate=dict(smoke='Original KDA/GRU/Mamba, 6 cells, first 3 saved cases',
        full='Only if all parity checks pass and projected CPU replay <=3600 seconds; otherwise report not completed, no expansion.'),
    claims=dict(allowed='Cross-configuration repeatability only.',
        excluded=['Sparse reward causality','Value defect from clearance spread','Larger gap implies better navigation','Delta mechanism innovation','Training variance estimated from two seeds']),
    reward_direction='Not cancelled; PBRS finite-budget benefit untested; no training authorized in this audit.',
    same_settings=dict(alpha=.3,selection='ttc5',actions=80,cases=[90000,90019],reserved_cases_untouched=[92000,92499]),
    evidence_scale='Observed cross-training-seed differences, not a formal noise threshold.',
))
print(out)
