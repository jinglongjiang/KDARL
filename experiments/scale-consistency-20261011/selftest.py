"""Check strict equivalence predicates and child-process polling without GPU work."""
import subprocess
import sys

import numpy as np

from common import BASE, compare_fields, save
from audit_existing_import import measure

row = dict(outcome='success', time=10., selection_trace_sha256='same',
           **{'return': .5}, minimum_clearance=.2)
assert not compare_fields(row, row.copy(), 1e-4)[0]
for key, value in [('outcome', 'timeout'), ('time', 10.25), ('selection_trace_sha256', 'changed')]:
    new = dict(row, **{key: value})
    assert key in compare_fields(row, new, 1e-4)[0]
for key in ('return', 'minimum_clearance'):
    new = dict(row, **{key: row[key] + .00005})
    assert not compare_fields(row, new, 1e-4)[0]
    assert key in compare_fields(row, new, 1e-6)[0]
    new[key] = row[key] + .0002
    assert key in compare_fields(row, new, 1e-4)[0]
proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.5)'])
assert proc.poll() is None
assert proc.wait(timeout=10) == 0 and proc.poll() == 0
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
result = measure(x, y, np.random.default_rng(0))
assert result['b'] == result['c'] == 1
assert result['disagreement'] == .5 and result['ratio'] == 1 and result['net_SR_pp'] == 0
ones = np.ones(10, dtype=bool)
assert measure(ones, ones, np.random.default_rng(0))['ratio'] is None
for path in BASE.glob('*.py'):
    compile(path.read_text(), str(path), 'exec')
save(BASE / 'selftest.json', dict(status='PASS', exact_fields_tested=3,
                                 numeric_tolerance_tested=True, process_alive_test=True,
                                 process_exited_test=True, paired_statistic_test=True,
                                 undefined_ceiling_ratio_test=True))
print('PASS: predicates, numeric tolerances, alive/exited polling, statistic and syntax')
