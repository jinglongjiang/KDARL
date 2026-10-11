"""Seal the resource-blocked preflight without deriving method efficacy."""
import json
import platform
from common import BASE, protocol, save, sha, verify

gate = json.loads((BASE / 'cost-gate.json').read_text())
done = json.loads((BASE / 'DONE.json').read_text())
assert gate['status'] == done['status'] == 'RESOURCE_LIMIT'
assert done['full_episodes'] == 0
assert not (BASE / 'per-case').exists()
checks = [json.loads((BASE / ('correctness-s%d.json' % seed)).read_text()) for seed in (419,443)]
smoke = sorted((BASE / 'smoke').glob('*/cell*.json'))
assert len(smoke) == 12
assert all(x['status'] == 'PASS' for x in checks)
assert all(len(json.loads(p.read_text())['episodes']) == 3 for p in smoke)
jobs = sorted((BASE / 'job-evidence').glob('*.json'))
assert len(jobs) == 14
assert all(json.loads(p.read_text())['exit_code'] == 0 for p in jobs)
summary = dict(
    status='RESOURCE_LIMIT', scientific_verdict='NOT_EVALUATED',
    correctness='PASS', smoke='PASS', smoke_episodes=36,
    formal_episodes_completed=0, formal_episodes_requested=6000, training_episodes=0,
    cost_gate=gate, correctness_by_seed=checks, checkpoints=protocol()['checkpoints'],
    frozen_files_verified=verify(), reserved92000_92499_untouched=True,
    source_project_modified=False, parameter_search=False,
    smoke_outcomes_not_used_for_method_acceptance=True,
    weights_not_original_optimized_seed419=True,
    weight_note='Both use fresh IL50+RL10000 history-study T24; original optimized cb2e checkpoint is not used',
    sources=dict(protocol=str(BASE/'protocol.json'), cost_gate=str(BASE/'cost-gate.json'),
                 job_exit_evidence=[str(p) for p in jobs], smoke_results=[str(p) for p in smoke]),
    failure_record='First runtime image lackedCcompiler beforemodelconstruction; archived. A subsequent manifestguard failure also archived. Correctness rerun only afterdevelimage andmanifestfix. Originalbudgetclock preserved.',
    limitations=['No AC3 vs TTC5/Goal navigationeffect estimate',
                 'No paired full-caseCI or rescue/damage conclusion',
                 'No novelty conclusion', '3case/cell throughput projection has sampling uncertainty'],
    recommendation='Need larger authorized wall budget or a faster available GPU; no automatic restart or sample reduction')
save(BASE / 'summary.json', summary)
save(BASE / 'input-verification.json', dict(status='PASS', files_verified=verify(),
                                           source_project_unchanged=True, weights_unchanged=True))
import torch
import fla
save(BASE/'environment-manifest.json',dict(python=platform.python_version(),platform=platform.platform(),
    torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
    fla_file=fla.__file__,image='nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04',
    existing_isolated_environment='/home/abc/temp/kda-vl-env',installed_or_upgraded_packages=False,
    home_mount='readonly except AC3workdir',cross_machine_numerical_parity='NOT_VERIFIED'))
files={}
for root in ('job-evidence','smoke','attempts'):
    for p in sorted((BASE/root).rglob('*')):
        if p.is_file():
            files[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
for name in ('RESULT.md','summary.json','protocol.json','frozen-manifest.json','cost-gate.json',
             'input-verification.json','environment-manifest.json','runtime-recovery.json','DONE.json'):
    p=BASE/name
    assert p.stat().st_size>0
    files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
save(BASE/'output-manifest.json',dict(files=files,status='SEALED_RESOURCE_LIMIT',
                                    evidence_is_not_full_experiment=True))
print(json.dumps(dict(status='RESOURCE_LIMIT',verdict='NOT_EVALUATED',frozen_verified=verify(),
                      report_bytes=(BASE/'RESULT.md').stat().st_size,sealed_files=len(files))),flush=True)
