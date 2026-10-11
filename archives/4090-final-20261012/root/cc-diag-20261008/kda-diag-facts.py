"""Source/config facts and a concrete smoothed-command reachability check."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path('/root/cc-diag-20261008')
spec = importlib.util.spec_from_file_location('eval_reference', root / 'eval6.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference.pin_project(root / 'code')
from crowd_nav.train import load_config
from crowd_nav.contracts import discrete_index_to_action, init_grid_from_cfg
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.utils.action import ActionXY

cfg = load_config(str(root / 'results' / 'kda.ini'))
grid = init_grid_from_cfg(cfg)
actions = [ActionXY(*discrete_index_to_action(i, **grid)) for i in range(80)]
speeds = [float(np.hypot(actions[i].vx, actions[i].vy)) for i in range(5)]
fake = SimpleNamespace(_phase='test', test_action_smoothing=.3,
                       _last_action=actions[1], action_space=actions)
commands = MambaRLPolicy.candidate_commands(fake)
minimum = min(float(np.hypot(c.vx, c.vy)) for c in commands)
assert minimum < .05, 'Expected a reachable command below stand threshold'
index = int(np.argmin([np.hypot(c.vx, c.vy) for c in commands]))
configuration = {s: dict(cfg.items(s)) for s in ('reward', 'eval_protocol', 'policy', 'mamba', 'train')}
source = {}
for p in sorted((root / 'code').rglob('*.py')):
    source[str(p.relative_to(root / 'code'))] = hashlib.sha256(p.read_bytes()).hexdigest()
output = {
    'configuration': configuration,
    'grid_speeds': speeds, 'include_stop': grid.get('include_stop'),
    'stand_branch_training_grid_unreachable': min(speeds) > .05,
    'stand_branch_evaluation_command_reachable': True,
    'counterexample': {'previous_actual_command': [actions[1].vx, actions[1].vy],
                       'candidate_grid_index': index,
                       'candidate_grid_command': [actions[index].vx, actions[index].vy],
                       'mixed_command': [commands[index].vx, commands[index].vy],
                       'mixed_speed': minimum},
    'index0_grid_command': [actions[0].vx, actions[0].vy],
    'stand_penalty_coefficient': cfg.getfloat('reward', 'stand_penalty'),
    'source_sha256': source,
}
(root / 'd9-facts.json').write_text(json.dumps(output, indent=2))
print(json.dumps({k: v for k, v in output.items() if k not in ('configuration', 'source_sha256')}))
