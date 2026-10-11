"""Freeze the amended 6000-episode protocol before any outcome evaluation."""
import configparser
import time
from common import BASE, PROJECT, HISTORY, save, sha


def main():
    assert not (BASE / 'protocol.json').exists()
    items = []
    for seed, root in (
        (419, '/home/abc/4090_backup_20261009/root/kda-history-20261009'),
        (443, '/home/abc/workspace/kda-history-replication-20261010'),
    ):
        from pathlib import Path
        root = Path(root)
        item = dict(seed=seed, weight=str(root / 'runs/kda-t24/rl_model_ep10000.pth'),
                    config=str(root / 'kda-t24.ini'), training='fresh IL50 + RL10000')
        item.update(weight_sha256=sha(item['weight']), config_sha256=sha(item['config']))
        cfg = configparser.ConfigParser()
        assert cfg.read(item['config'])
        assert cfg.getint('temporal', 't') == cfg.getint('buffer', 'seq_len') == 24
        # The419 run supplies seed through its launcher, not its config.
        import json
        training_protocol = root / 'protocol.json'
        assert json.loads(training_protocol.read_text())['seed'] == seed
        if cfg.has_option('train', 'seed'):
            assert cfg.getint('train', 'seed') == seed
        item['training_protocol'] = str(training_protocol)
        assert cfg.getint('sim', 'human_num') == 5
        assert cfg.getfloat('eval_protocol', 'action_smoothing') == .3
        assert cfg.getfloat('eval_protocol', 'risk_lambda') == .8
        assert cfg.getfloat('eval_protocol', 'safety_margin') == .2
        items.append(item)
    save(BASE / 'protocol.json', dict(
        task='AC3_ACTION_COVERAGE_SELECTION', created=time.time(), checkpoints=items,
        amendment='Latest user instruction supersedes downloaded384-episode plan:6000episodes.',
        rules=['ttc5', 'goal-cv1s', 'ac3'], cells=[4, 5], case_base=90000, n_cases=500,
        total_episodes=6000, training_episodes=0, reserved92000_92499_untouched=True,
        selection_anchor='true current state once; same identities for current and80successors',
        selection_budget=5, alpha=.3, risk_lambda=.8, filter_margin=.2, dt=.25,
        horizon=1., max_episode_steps=201, gamma=.99, time_limit=50.,
        goal_cv='robot v_pref along current goal direction; same-time closest approach over1s; lowest5clearances',
        ac3='Downloaded reference risk saturation coverage; canonical TTC fallback/zero-gain fill',
        ttc_rounding='Existing TTC5 robot float32 roundtrip and np.argsort; preserve source-order selectedIDs',
        all_unsafe='AC3 returns exact current TTC5; parent fallback unchanged',
        five_person_bypass='all3selectors original5; identical tokens/values/actions/trajectories',
        forbidden=['training', 'project edits', 'future features', 'reward/filter/grid/network changes',
                   'threshold/horizon search', 'reserved confirmation cases', 'additionalseeds'],
        primary='original task discountedReturn', secondary='SR',
        contrasts=[['ac3','ttc5'],['ac3','goal-cv1s'],['goal-cv1s','ttc5']],
        inference='caseID cluster pairedbootstrap, retain2cells and2seeds withincluster;10000draws;95%two-sidedCI',
        intervals='descriptive; same developmentblock, not fresh confirmation; n_training_seeds=2',
        decision=dict(require_return_positive_vs_both_each_seed=True,
                      require_SR_nonnegative_vs_both_each_seed=True,
                      require_collision_count_nonincrease_vs_both_each_seed=True,
                      safety_note='Conservative explicit implementation of no unacceptablecollisionincrease; not a significance claim',
                      positive_label='PROTOTYPE_ENTRY_NOT_CONFIRMED', negative_label='REJECT',
                      no_parameter_search_after_results=True),
        runtime='local3060 isolated existing torch2.7.1 env in Ubuntu22readonly container; no packageupgrades',
        cost_gate=dict(budget_seconds=7200, starts='first correctness worker launch',
                      smoke_episodes=36, max_workers=3,
                      projection='full6000episodes from parallel smoke throughput, plus measuredchecks and120sanalysisreserve',
                      insufficient='RESOURCE_LIMIT; do not reduce500cases or startfull'),
        missing_prior_omission_measurement='11737step/23.48% user-provided context, not independently verified by this protocol',
        limitations=['Coveragehit is not navigationbenefit', 'CV1s is not learned interactionprediction',
                     'Selectionchanges are distribution shifts with frozenweights',
                     'Outcome-associated補選 statistics do not establish isolatedstepcausality',
                     'No noveltyclaim; separate precisionprioraudit only if prototype passes']))
    files = {}
    for item in items:
        for key in ('weight', 'config', 'training_protocol'):
            files[item[key]] = sha(item[key])
    for root in (PROJECT / 'crowd_nav', PROJECT / 'crowd_sim', PROJECT / 'tools',
                 __import__('pathlib').Path('/home/abc/workspace/kda-deps/flash-linear-attention/fla')):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.ini', '.config'):
                files[str(path)] = sha(path)
    for path in [HISTORY, BASE / 'protocol.json', *sorted(BASE.glob('*.py')),
                 __import__('pathlib').Path('/home/abc/Downloads/action_coverage_selection.py'),
                 __import__('pathlib').Path('/home/abc/Downloads/ONE_CANDIDATE_SPEC.md')]:
        files[str(path)] = sha(path)
    save(BASE / 'frozen-manifest.json', dict(files=files, weights=items))
    print(dict(status='FROZEN', files=len(files), checkpoints=items), flush=True)


if __name__ == '__main__':
    main()
