"""Regression: value-lookahead actions must not be dropped by replay."""

import configparser
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowd_nav.contracts import init_grid_from_cfg
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState


class TrainingContract(unittest.TestCase):
    def test_fixed_budget_config_wrapper(self):
        from crowd_nav.train import TrainConfig, _run_policy_il_pretrain
        torch.set_num_threads(2)
        cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        cfg.read([str(ROOT / 'crowd_nav/configs' / f'{name}.config')
                  for name in ('env', 'policy', 'train')])
        cfg.set('mamba', 'temporal_backbone', 'gru')
        cfg.set('train', 'il_fixed_budget', 'true')
        net = MambaRLPolicy(cfg, device='cpu')
        state = JointState(FullState(0, -4, 0, 0, .3, 0, 4, 1, 0),
                           [ObservableState(3 + i, i, 0, 0, .3) for i in range(5)])
        with tempfile.TemporaryDirectory() as directory:
            wrapper = TrainConfig(cfg, directory)
            self.assertTrue(wrapper.il_fixed_budget)
            path = Path(directory) / 'il.pth'
            trajectory = ([state.to_array()] * 9, [(0., 1.)] * 9, [0.] * 8 + [1.])
            _run_policy_il_pretrain(net, torch.device('cpu'), 24, 2, 8, str(path), None,
                                    gamma=.99, success_trajs=[(trajectory, {})], config=wrapper)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['meta']['epochs'], 2)
            self.assertEqual(checkpoint['meta']['objective'], 'value_regression')
            self.assertTrue(np.isfinite(checkpoint['meta']['value_loss']))

    def test_token_conversion_commutes_with_windowing(self):
        from crowd_nav.train import make_il_token_windows
        from crowd_nav.contracts import joint34_to_tokens
        rng = np.random.default_rng(419)
        for length in (1, 3, 24, 31):
            raw = rng.normal(size=(length, 34)).astype(np.float32)
            for seq_len in (1, 3, 24):
                windows = []
                for t in range(length):
                    window = raw[max(0, t - seq_len + 1):t + 1]
                    window = np.concatenate([np.repeat(window[:1], seq_len - len(window), axis=0), window])
                    windows.append(window)
                expected = joint34_to_tokens(np.asarray(windows).reshape(-1, 34))
                expected = expected.reshape(length, seq_len, 8, 13).numpy()
                np.testing.assert_array_equal(make_il_token_windows(raw, seq_len), expected)

    def test_value_policy_records_real_grid_index_and_mc_targets(self):
        torch.set_num_threads(2)
        cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        cfg.read([str(ROOT / 'crowd_nav/configs' / f'{name}.config')
                  for name in ('env', 'policy', 'train')])
        env_cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'))
        env_cfg.read(str(ROOT / 'crowd_nav/configs/env.config'))
        init_grid_from_cfg(env_cfg)
        cfg.set('mamba', 'temporal_backbone', 'gru')
        net = MambaRLPolicy(cfg, device='cpu')
        net.use_sarl_predict = True
        net.set_phase('train')
        state = JointState(FullState(0, -4, 0, 0, .3, 0, 4, 1, 0),
                           [ObservableState(3 + i, i, 0, 0, .3) for i in range(5)])
        actions, indices = [], []
        with torch.no_grad():
            for epsilon in (1., 0., 0.):
                net.epsilon = epsilon
                action, index = net.act(state)
                self.assertIsInstance(index, int)
                self.assertEqual(tuple(action), tuple(net.action_space[index]))
                actions.append(tuple(action))
                indices.append(index)
            action, index = net.predict(state, return_idx=True)
            self.assertEqual(tuple(action), tuple(net.action_space[index]))
        replay = ReplayBufferIQL(capacity=16, seq_len=24, gamma=.99)
        replay.push_episode({'states': [state.to_array()] * 3, 'actions': actions,
                             'action_indices': indices, 'rewards': [0., 0., 1.],
                             'dones': [False, False, True]})
        self.assertEqual(len(replay), 3)
        np.testing.assert_allclose(replay.returns[:3], [.99 ** 2, .99, 1.])
        np.testing.assert_array_equal(replay.action_indices[:3], indices)


if __name__ == '__main__':
    unittest.main()
