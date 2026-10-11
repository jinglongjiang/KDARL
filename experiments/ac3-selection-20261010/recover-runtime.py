"""Archive failed pre-model runtime attempt; preserve original manifest and clock."""
from common import BASE, save, sha
import json

archive = BASE / 'attempts/runtime-no-compiler'
archive.mkdir(parents=True, exist_ok=True)
for name in ('ERROR.json','status.json','logs','job-evidence','frozen-manifest.json'):
    path=BASE/name
    if path.exists():
        destination=archive/name
        if destination.exists():
            destination=archive/('recovery-guard-'+name)
        assert not destination.exists()
        path.rename(destination)
original=json.loads((archive/'frozen-manifest.json').read_text())
files=original['files']
changed=[p for p,h in files.items() if sha(p)!=h]
assert changed==[str(BASE/'run.py')], changed
files[str(BASE/'run.py')]=sha(BASE/'run.py')
save(BASE/'frozen-manifest.json', original)
save(BASE/'runtime-recovery.json', dict(
    cause='runtimeUbuntu22image has noCcompiler; Tritonimport failed before modelconstruction',
    fix='Use preexisting develUbuntu22image; no packagesor methodchanges',
    code_change='Controller restart preserves first-start two-hourdeadline',
    original_manifest=str(archive/'frozen-manifest.json'), changed_files=changed,
    no_outcome_data_seen=True, clock_reset=False))
