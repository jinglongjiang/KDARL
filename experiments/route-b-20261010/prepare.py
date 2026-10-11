"""Freeze inputs and semantics before inference; does not construct a model."""
import configparser
import json
from pathlib import Path
import time
from common import BASE, WORK, PROJECT, save, sha


def main():
    assert not (BASE / 'protocol.json').exists(), 'Do not overwrite frozen protocol'
    items = []
    for seed, root in [(419, Path('/home/abc/4090_backup_20261009/root/kda-history-20261009')),
                       (443, WORK / 'kda-history-replication-20261010')]:
        for arm in ('kda', 'gru'):
            for length in (12, 24):
                name = '%s-s%d-t%d' % (arm, seed, length)
                items.append(dict(id=name, backbone=arm, seed=seed, length=length,
                    weight=str(root / 'runs' / ('%s-t%d' % (arm, length)) / 'rl_model_ep10000.pth'),
                    config=str(root / ('%s-t%d.ini' % (arm, length))),
                    training='fresh IL50+RL10000', origin='history-length study'))
    artifact = PROJECT / 'artifacts/vl-optimized-20261007'
    for arm in ('kda', 'gru', 'mamba'):
        items.append(dict(id=arm+'-original-t24', backbone=arm, seed=419, length=24,
            weight=str(artifact / arm / 'rl_model_ep10000.pth'),
            config=str(artifact / (arm+'.ini')), training='reused IL50+fresh RL10000',
            origin='original optimized run'))
    frozen = {}
    for item in items:
        for key in ('weight', 'config'):
            p = Path(item[key]); assert p.is_file() and p.stat().st_size > 0, p
            frozen[str(p)] = sha(p)
        item['weight_sha256'] = frozen[item['weight']]
        item['config_sha256'] = frozen[item['config']]
        cfg = configparser.ConfigParser(); assert cfg.read(item['config'])
        assert cfg.getint('temporal', 't') == cfg.getint('buffer', 'seq_len') == item['length']
        assert cfg.getfloat('eval_protocol', 'action_smoothing') == .3
        assert cfg.getfloat('eval_protocol', 'risk_lambda') == .8
    assert len(items) == 11 and len({x['weight_sha256'] for x in items}) == 11
    for root in (PROJECT / 'crowd_nav', PROJECT / 'crowd_sim', PROJECT / 'tools'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.ini', '.config'):
                frozen[str(p)] = sha(p)
    for p in sorted((WORK / 'kda-deps/flash-linear-attention/fla').rglob('*.py')):
        frozen[str(p)] = sha(p)
    reference = WORK / 'kda-history-20261009/evaluate-history.py'
    frozen[str(reference)] = sha(reference)
    for p in sorted(BASE.glob('*.py')):
        frozen[str(p)] = sha(p)
    save(BASE / 'protocol.json', dict(task='ROUTE_B_EVIDENCE_CONSOLIDATION',
        created=time.time(), train_episodes=0, optimizer_steps=0, checkpoints=items,
        project=str(PROJECT), original_projects_read_only=True,
        runtime='temporary local3060 container, existing isolated torch2.7.1 env; no package changes',
        conditions=dict(alpha=.3, risk_lambda=.8, safety_margin=.2, actions=80, selection='ttc5',
            selection_anchor='once at true current robot state; shared by all candidates',
            horizon_seconds=.25, case_base=90000, smoke_cases_per_cell=3,
            full_cases_per_cell=20, optional_expansion=False,
            sample_choice='20/cell is frozen; do not expand based on observed margin results',
            common_reference='kda-original-t24',
            common_windows='same reference prefix/successor tokens and commands; T12 uses final12 frames'),
        metrics=['unmasked successor V top1-top2', 'gap/abs(top1 V), zero-scale flagged',
            'raw r+gammaV gap separately (not V gap)',
            'top5-V clearance range, min/max and individual candidates',
            'actual next-position endpoint clearance',
            'environment old-human-velocity same-time swept clearance',
            'actual-update same-time swept clearance'],
        clearance_bins=['<0', '[0,0.2)', '[0.2,0.4)', '[0.4,0.8)', '>=0.8'],
        interval_unit='episode clustered; descriptive bootstrap, not independent steps or11 independent seeds',
        decision=dict(gap_median_range=[1e-4, 1e-2], clearance_spread_median_range_m=[.01, 1.],
            range_definition='broad operational order-of-magnitude screen, not proof of equivalence',
            outcome='heterogeneity across checkpoints downgrades common-configuration claim',
            causal_limit='shared training objective causation cannot be isolated without an objective intervention',
            interpretation='V gap is not Q action gap; one-step clearance is not long-run action quality'),
        statistical_limits=['No small gap proves value failure',
            'flat clearance-bin spread is descriptive, not statistical independence',
            'two seed absolute differences are observed variation, not an estimated noise floor',
            'same checkpoint interventions can still include finite-case and runtime uncertainty'],
        reserved_cases_92000_92499_untouched=True,
        forbidden=['training', 'original-project edits', 'new outcome benchmark beyond P1',
                   'new action/filter/reward/model', '4090 usage']))
    frozen[str(BASE / 'protocol.json')] = sha(BASE / 'protocol.json')
    save(BASE / 'frozen-manifest.json', dict(files=frozen, checkpoints=items,
        weight_count=11, policy_source_pinned=True, timestamp=time.time()))
    print(json.dumps(dict(status='PASS', weights=len(items), frozen_files=len(frozen))), flush=True)


if __name__ == '__main__':
    main()
