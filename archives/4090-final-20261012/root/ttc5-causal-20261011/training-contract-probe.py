"""Read-only unit reproduction of exploration/history and deployment contracts."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from types import MethodType
import torch
import audit as A

BASE=Path(__file__).resolve().parent


def main():
    output=BASE/'work/training-contract-probe.json'
    assert not output.exists()
    result=[]
    with torch.no_grad():
        for model in ('kda-s419-t24','gru-s419-t24'):
            _,_,net=A.load(model)
            from crowd_nav.policy.mamba_rl import MambaRLPolicy
            from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
            from crowd_sim.envs.utils.action import ActionXY
            # The original training path has no evaluation-only TTC patch.
            net._build_joint_state_34=MethodType(MambaRLPolicy._build_joint_state_34,net)
            net.score_sarl_candidates=MethodType(MambaRLPolicy.score_sarl_candidates,net)
            ep=next(A.rows(BASE/'work'/('baseline-%s-cell0.jsonl.gz'%model)))
            state_rows=ep['steps'][:3]
            trials=[]
            for eps in (1.,0.):
                net.reset_episode_stats();net.set_phase('train');net.set_epsilon(eps)
                lengths=[]
                for row in state_rows:
                    state=JointState(FullState(*row['robot']),[ObservableState(*h) for h in row['humans']])
                    before=len(net._history)
                    net.predict(state)
                    lengths.append([before,len(net._history)])
                assert lengths==([[0,0]]*3 if eps else [[0,1],[1,2],[2,3]])
                trials.append(dict(forced_epsilon=eps,history_lengths=lengths,provided_real_frames=3))
            commands={}
            net._last_action=ActionXY(.5,0.)
            for phase in ('train','test'):
                net.set_phase(phase)
                commands[phase]=[[a.vx,a.vy] for a in net.candidate_commands()]
            assert len(commands['train'])==len(commands['test'])==80
            for raw,actual in zip(commands['train'],commands['test']):
                assert actual==[.3*.5+.7*raw[0],.7*raw[1]]
            item=next(m for m in A.P['models'] if m['id']==model)
            actual_sha=A.REF.sha(item['weight'])
            assert actual_sha==item['weight_sha256']
            result.append(dict(model=model,weight_sha256=actual_sha,
                               trials=trials,candidate_commands=commands,
                               original_checkpoint_unchanged=True))
    data=dict(status='PASS',unit_results=result,script_sha256=A.REF.sha(__file__),
              scope='No optimizer, no environment rollout, no train data written. Epsilon forced only in isolated unit test. Original training build/score methods, not evaluation-only TTC wrappers.',
              verdict='Exploration omits current frame from online policy history. Full replay-window history still stores every real frame. Navigation loss of this mismatch is not isolated here.',
              source_references={'random_early_return':'mamba_rl.py:765-768',
                                 'history_append':'mamba_rl.py:777',
                                 'all_frames_recorded':'explorer.py:466',
                                 'episode_into_MC_buffer':'train.py:2568-2600',
                                 'eval_only_smoothing':'mamba_rl.py:785-791',
                                 'eval_only_mask_and_risk':'mamba_rl.py:892-902'})
    A.save(output,data)
    print(json.dumps(dict(status='PASS',models=len(result),file=str(output))),flush=True)


if __name__=='__main__':main()
