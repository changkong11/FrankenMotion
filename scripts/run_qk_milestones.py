"""Bounded train/evaluate sequence at epochs 699 and 899."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(os.environ.get('FM_MILESTONE_WORK','logs/qk_milestones_699_899_v1'))

def main():
    os.chdir(Path(__file__).resolve().parents[1]);WORK.mkdir(exist_ok=False)
    env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**row):
            row['time']=time.time();line=json.dumps(row);status.write(line+'\n');status.flush();print(line,flush=True)
        def run(label,module):
            emit(event='started',stage=label)
            with (WORK/(label+'.out')).open('x') as out:
                p=subprocess.run([sys.executable,'-u','-m',module],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            emit(event='exited',stage=label,code=p.returncode)
            if p.returncode:raise SystemExit(p.returncode)
        source=os.environ.get('FM_MILESTONE_SOURCE','outputs_amass/formal_qk_g2p_340_499_v1')
        stages=json.loads(os.environ.get('FM_MILESTONE_STAGES','[[500,699],[700,899]]'))
        for start,end in stages:
            emit(event='waiting_for_gpu',epoch=end,minimum_free_mib=36500)
            while True:
                lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
                available=[(int(l.split(',')[1]),l.split(',')[0].strip()) for l in lines if int(l.split(',')[1])>=36500]
                if available:break
                time.sleep(60)
            env['CUDA_VISIBLE_DEVICES']=max(available)[1]
            target=f'outputs_amass/formal_qk_g2p_{start}_{end}_v1'
            evaluation=f'logs/formal_qk{end}_validation_v1'
            env.update(FM_EXTEND_SOURCE=source,FM_EXTEND_TARGET=target,FM_EXTEND_START=str(start),FM_EXTEND_END=str(end),FM_EVAL_SOURCE=target,FM_EVAL_WORK=evaluation,FM_EVAL_EPOCH=str(end))
            run(f'train_{end}','scripts.extend_formal_qk')
            run(f'eval_{end}','scripts.evaluate_qk499')
            import scripts.audit_motion_dynamics as dynamics
            import scripts.audit_generated_jerk as jerk
            reference=Path('logs/formal_qk499_validation_v1/formal_g2p499_fullgraph/generations/t2m/val_model_20260929_140930')
            generated=list((Path(evaluation)/f'formal_g2p{end}_fullgraph/generations/t2m').glob('val_model_*'))
            assert len(generated)==1
            dynamics.BASE=jerk.BASE=reference;dynamics.NEW=jerk.NEW=generated[0]
            import shutil
            dynamics.main();shutil.copy2('logs/motion_dynamics_audit.json',WORK/f'dynamics_{end}_vs499.json')
            jerk.main();shutil.copy2('logs/generated_jerk_audit.json',WORK/f'jerk_{end}_vs499.json')
            emit(event='milestone_complete',epoch=end)
            source=target
        emit(event='pipeline_complete')

if __name__=='__main__':main()
