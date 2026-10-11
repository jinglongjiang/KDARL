# -*- coding: utf-8 -*-
"""只对 kda 臂调用他们自己的 evaluate(); 不改他们的代码, 不碰 protocol.json。"""
import argparse, json, sys, types
from pathlib import Path
import torch
sys.path.insert(0, '/root/kda-vl-3000-20261006/code')
sys.path.insert(0, '/root/kda-vl-3000-20261006/code/tools')
import run_vl_minimal as R

ap = argparse.ArgumentParser()
ap.add_argument('--output', required=True)
ap.add_argument('--arm', default='kda')
ap.add_argument('--seed', type=int, default=419)
ap.add_argument('--cells', type=int, default=0)   # 0=全部
ap.add_argument('--cases', type=int, default=0)   # 0=全部
a = ap.parse_args()
torch.set_num_threads(2)
proto = json.loads(Path('/root/kda-vl-3000-20261006/results/protocol.json').read_text())
if a.cells: proto['eval_cells'] = proto['eval_cells'][:a.cells]
if a.cases: proto['eval_cases'] = proto['eval_cases'][:a.cases]
print(f"[driver] arm={a.arm} out={a.output} cells={len(proto['eval_cells'])} "
      f"cases={len(proto['eval_cases'])} -> {len(proto['eval_cells'])*len(proto['eval_cases'])} episodes", flush=True)
args = types.SimpleNamespace(output=a.output, seed=a.seed, dataset=None,
                            source_commit=None, case_start=proto['eval_cases'][0])
R.evaluate(args, a.arm, proto)
print("[driver] done", flush=True)
