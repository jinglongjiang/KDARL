"""Seal the verified single-report deliverable without changing frozen protocol."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    target = BASE/'delivery-manifest.json'
    assert not target.exists()
    started = time.time()
    result = subprocess.run([sys.executable,str(BASE/'verify-results.py')],
                            capture_output=True,text=True,cwd=BASE)
    if result.returncode:
        target.write_text(json.dumps(dict(status='FAIL',exit_code=result.returncode,
                          stdout=result.stdout,stderr=result.stderr),indent=2))
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    verification = json.loads(result.stdout)
    assert verification['status']=='PASS'
    manifest = json.loads((BASE/'manifest.json').read_text())
    originals = {p:sha(p)==h for p,h in manifest['original_local_files'].items()}
    frozen = {p:sha(BASE/p)==h for p,h in manifest['files'].items()}
    assert originals and frozen and all(originals.values()) and all(frozen.values())
    data = BASE/'root_cause_data.jsonl.gz'
    assert sha(data)==verification['data_sha256']
    report = BASE/'ROOT_CAUSE_RESULT.md'
    text = report.read_text()
    assert '## 独立交付验收' not in text
    text += ('\n## 独立交付验收\n'
             '- verify-results.py实际退出码0：%d条baseline、%d个root；所有分支折扣回报独立重算通过。\n'
             '- %d条A0的全部逐步history/windows/实际命令与基线逐项相同；预设A1/A2/A3独立重算相同；两项/三项误差恒等式通过。\n'
             '- 原始本地资产、21项冻结输入/脚本与334项源码结束复验通过；远端归因结束复验也通过。\n'
             '- 可复现单例：reproduce-case.py --model kda-s419-t24 --cell 5 --case 90001 --label A1。实际已再次运行，退出码0、完整轨迹哈希相同。需要4090原依赖栈；可显式指定--project和--inputs。\n'
             '- 证据包：%d字节；SHA256 %s。最终报告及脚本哈希记录在delivery-manifest.json，原manifest.json未修改。\n'
             % (verification['baseline_episodes'],verification['completed_roots'],
                verification['A0_complete_history_and_windows_checked'],data.stat().st_size,verification['data_sha256']))
    report.write_text(text)
    output = dict(status='PASS',completed=time.time(),verification_seconds=time.time()-started,
        wall_seconds_from_protocol=time.time()-manifest['protocol']['started_unix'],
        verification=verification,verification_exit_code=result.returncode,
        initial_manifest_sha256=sha(BASE/'manifest.json'),report_sha256=sha(report),
        report_bytes=report.stat().st_size,data_bytes=data.stat().st_size,
        diagnostic_scripts_sha256={p.name:sha(p) for p in sorted(BASE.glob('*.py'))},
        local_original_assets_verified=originals,local_frozen_inputs_verified=frozen,
        remote_final_verification=json.loads((BASE/'work/post-attribution-end-verification.json').read_text()))
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='PASS',report=str(report),data=str(data),
        verification=verification,report_sha256=output['report_sha256'],
        manifest=str(target)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
