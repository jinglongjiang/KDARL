"""CPU algebra checks; no project imports or learning."""
import json
from pathlib import Path

gamma = .99
reward = [0., 0.]
value = [.6, .5]
phi_current = .25
phi_next = [0., 1.]
original = [r + gamma*v for r,v in zip(reward,value)]
shaped_reward = [r + gamma*p - phi_current for r,p in zip(reward,phi_next)]
consistent = [r + gamma*(v-p) for r,v,p in zip(shaped_reward,value,phi_next)]
reward_only = [r + gamma*v for r,v in zip(shaped_reward,value)]
argmax = lambda xs: max(range(len(xs)), key=lambda i: xs[i])
assert max(abs(a-(b-phi_current)) for a,b in zip(consistent,original)) < 1e-12
assert argmax(consistent) == argmax(original) == 0
assert argmax(reward_only) == 1
out = dict(status='已验证通过', gamma=gamma, original=original,
           consistent_shaping=consistent, reward_only_shaping=reward_only,
           original_argmax=argmax(original), consistent_argmax=argmax(consistent),
           reward_only_argmax=argmax(reward_only), training=False,
           scope='Algebra, not empirical navigation improvement')
path = Path(__file__).parent/'pbrs-check.json'
path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,ensure_ascii=False))
