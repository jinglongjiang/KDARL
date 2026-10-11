"""Engineering-only fixtures, never navigation evidence."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pipeline', BASE / 'run-diagnostics.py')
pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)

with tempfile.TemporaryDirectory(prefix='controller-unit-', dir=BASE) as directory:
    pipeline.BASE = Path(directory); pipeline.verify_freeze = lambda: None
    live = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(.1)'])
    assert pipeline.alive(live.pid); live.wait(); assert not pipeline.alive(live.pid)
    out = pipeline.BASE / 'fixture.json'
    fixture = dict(status='已验证通过', grid_actions=80, candidate_actions=80, selection='ttc5',
        checks=dict(selection=1, scores=1, history=1), episodes=[dict(outcome='success', return_value=0.)])
    code = 'import json,pathlib; pathlib.Path(%r).write_text(%r); print("ENGINEERING FIXTURE ONLY")' % (str(out), json.dumps(fixture))
    task = dict(command=[sys.executable, '-B', '-c', code], output=out, name='nested/smoke-fixture', expected_n=1, candidate_count=80)
    pipeline.run_jobs([task], 'unit-fixture')
    assert (pipeline.BASE / 'logs/nested/smoke-fixture.log').exists()
    task = dict(command=[sys.executable, '-B', '-c', 'raise SystemExit(7)'], output=pipeline.BASE / 'never.json',
        name='nested/intentional-nonzero', expected_n=1, candidate_count=80)
    try:
        pipeline.run_jobs([task], 'unit-fixture-negative')
    except AssertionError:
        assert json.loads((pipeline.BASE / 'job-evidence/nested/intentional-nonzero.json').read_text())['exit_code'] == 7
    else:
        raise AssertionError('nonzero fixture was not rejected')
result = dict(status='已验证通过', engineering_only=True, nested_log_directory=True,
    subprocess_metadata_validation=True, nonzero_exit_rejected=True, pid_running_and_finished=True)
(BASE / 'controller-unit-tests.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result), flush=True)
