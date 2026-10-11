"""Read original logs only; extract inspectable second-round cost anchors."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


def analyze(path):
    lines = path.read_text().splitlines()
    def first(needle):
        index = next(i for i,line in enumerate(lines) if needle in line)
        raw = lines[index]
        stamp = re.match(r'\[(\d\d-\d\d \d\d:\d\d:\d\d)\]',raw).group(1)
        dt = datetime.strptime('2026-'+stamp,'%Y-%m-%d %H:%M:%S')
        return {'line':index+1,'timestamp':dt.isoformat(),'raw':raw},dt
    start,t0 = first('[TENSORBOARD] Fresh events')
    il_end,ti = first('[IL-BC-EVAL] epoch=50 quick_eval')
    rl_begin,tr = first('[RL-EP-0001]')
    rl_end,te = first('[REPLAY-STORE] ok=10000')
    end,tf = first('[MAIN-COMPLETE]')
    frames = int(re.search(r' steps=(\d+)',rl_end['raw']).group(1))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'start':start,'IL50_epoch_end':il_end,'RL_first_completed_episode':rl_begin,
        'RL_10000_completed_episodes':rl_end,'saved_and_completed':end,
        'total_wall_seconds':(tf-t0).total_seconds(),'IL_start_to_epoch50_seconds':(ti-t0).total_seconds(),
        'RL_ep1_to_ep10000_seconds':(te-tr).total_seconds(),'RL_stored_reward_count':frames,
        'old_ttc5_log_seconds_per_reward':(te-tr).total_seconds()/frames,
        'scope':'Original seed443 schedule max4 training workers sharing GPU; measured wall, not isolated GPU utilization. No full-N20 throughput measured.'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/home/abc/workspace/kda-history-replication-20261010/runs')
    args = p.parse_args()
    print(json.dumps({'authorized_training':False,'observations':[
        analyze(Path(args.root)/name/'train.log') for name in ['gru-t24','kda-t24']],
        'cost_contract':'N5 R1/R2 redundant if full forward+gradient+optimizer parity proven; 3 unique trainings per seed, 6 GRU runs, 8 evaluation cells. No full-N20 cost promise.'},indent=2))
