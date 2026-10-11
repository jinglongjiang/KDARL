"""Isolated deterministic contract tests; no optimizer or environment rollout."""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--before', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.project)
    import crowd_nav, crowd_sim
    from crowd_nav import contracts
    import crowd_nav.policy.mamba_rl as current
    from crowd_nav.train import load_config
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    for mod in (crowd_nav, crowd_sim, current):
        Path(mod.__file__).resolve().relative_to(Path(a.project).resolve())
    spec = importlib.util.spec_from_file_location('original_policy', a.before)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cfg = load_config(a.config)
    cfg.set('mamba', 'temporal_backbone', 'gru')
    contracts.init_grid_from_cfg(cfg)
    assert contracts.grid_action_dim(contracts.GRID) == 80
    torch.set_num_threads(1)

    def create(cls, length=24):
        torch.manual_seed(419)
        n = cls(cfg, device='cpu')
        n.eval()
        n.seq_len = length
        n._history = deque(maxlen=length)
        n.build_action_space(1.)
        # Actual scorer and token/history code; deterministic cheap value stand-in.
        n.forward_value = MethodType(lambda self, x: x[:, -1, 0, 0] * .13 + .7, n)
        return n

    def state(k, x=0., goal=4., near=False):
        r = FullState(x + k*.007, -4.+k*.01, .1, .2, .3, 0., goal, 1., 0.)
        h = [ObservableState(20.+j, 20., 0., 0., .3) for j in range(5)]
        if near:
            h[0] = ObservableState(r.px + .65, r.py, -.1, 0., .3)
        return JointState(r, h)

    trials = []
    for length in (12, 24):
        for mode in ('random', 'greedy', 'alternating'):
            n = create(current.MambaRLPolicy, length)
            n.set_phase('train')
            n.epsilon = .5
            expected = deque(maxlen=length)
            for k in range(length + 3):
                s = state(k)
                token = contracts._batch_joint34_to_tokens_vectorized(
                    n._build_joint_state_34(s.self_state, s.human_states).reshape(1, -1))[0]
                random = mode == 'random' or (mode == 'alternating' and k%2 == 0)
                with patch('numpy.random.random', return_value=0. if random else .9), \
                     patch('numpy.random.choice', return_value=3):
                    n.predict_sarl_style(s)
                expected.append(token)
                assert len(n._history) == len(expected)
                assert all(np.array_equal(u, v) for u, v in zip(n._history, expected))
                w = list(n._history)
                w = [w[0]] * (length-len(w)) + w
                e = list(expected)
                e = [e[0]] * (length-len(e)) + e
                assert np.array_equal(w, e)
            n.reset_episode_stats()
            assert len(n._history) == 0
            trials.append(dict(length=length, mode=mode, observations=length+3, status='PASS'))

    n0, n1 = create(old.MambaRLPolicy), create(current.MambaRLPolicy)
    for n in (n0,n1):
        n.set_phase('test')
        n.epsilon = 0.
    parity = 0
    for k in range(10):
        s = state(k)
        o0, o1 = n0.score_sarl_candidates(s), n1.score_sarl_candidates(s)
        assert not o1['predicted_terminal'].any()
        for key in ('windows', 'values', 'rewards', 'clearance', 'raw_scores', 'scores'):
            assert torch.equal(o0[key], o1[key]), (k,key)
        assert np.array_equal(o0['current_token'], o1['current_token'])
        u0, u1 = n0.predict_sarl_style(s), n1.predict_sarl_style(s)
        assert u0 == u1
        assert all(np.array_equal(u,v) for u,v in zip(n0._history,n1._history))
        parity += 1

    terminals = []
    for label, s in [('goal', state(0, goal=-3.8)), ('collision', state(0, near=True))]:
        n1.reset_episode_stats()
        o = n1.score_sarl_candidates(s)
        mask = o['predicted_terminal']
        if label == 'collision':
            # Put one human exactly on the endpoint of a native candidate.
            cmd = n1.action_space[-1]
            s.human_states[0] = ObservableState(s.self_state.px + cmd.vx*.25,
                s.self_state.py + cmd.vy*.25, 0., 0., .3)
            o = n1.score_sarl_candidates(s)
            mask = o['predicted_terminal']
        assert mask.any(), label
        assert torch.equal(o['raw_scores'][mask], o['rewards'][mask])
        legacy = n1._score_sarl_candidates_legacy(s)
        assert torch.equal(o['windows'], legacy['windows'])
        assert torch.equal(mask, legacy['predicted_terminal'])
        assert torch.allclose(o['raw_scores'], legacy['raw_scores'], atol=2e-15, rtol=0.)
        for mode in ('reward_only','value_only','full'):
            n1.lookahead_ablation_mode = mode
            om = n1.score_sarl_candidates(s)
            if mode == 'value_only':
                assert torch.equal(om['raw_scores'][mask], torch.zeros_like(om['raw_scores'][mask]))
        n1.lookahead_ablation_mode = 'full'
        terminals.append(dict(condition=label, detected=int(mask.sum()), status='PASS'))
    n1.set_env(SimpleNamespace(global_time=50., time_limit=50.))
    o = n1.score_sarl_candidates(state(0))
    assert o['predicted_terminal'].all()
    assert torch.equal(o['raw_scores'],o['rewards'])
    terminals.append(dict(condition='known_timeout',detected=80,status='PASS'))
    data = dict(status='PASS', history_trials=trials, epsilon_zero_bitwise_states=parity,
                terminal_tests=terminals, grid=80, torch=torch.__version__,
                scope='CPU unit tests, stand-in value only. No optimizer; physical CV/reward/goal contract differences unchanged.')
    Path(a.out).write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
