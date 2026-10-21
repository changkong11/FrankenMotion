"""Matched one-run full validation evaluation of historical checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from omegaconf import OmegaConf
from src.tools.inference import load_diffusion

WORK=Path('logs/matched_historical_v1')

def main():
    os.chdir(Path(__file__).resolve().parents[1]);torch.set_num_threads(4)
    os.environ.update(FM_EVAL_TEXT_MODEL_PATH=str(Path('pretrained/distilbert-base-uncased').resolve()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**r):
            r['time']=time.time();line=json.dumps(r);status.write(line+'\n');status.flush();print(line,flush=True)
        def run(label,args):
            emit(event='started',stage=label)
            with (WORK/(label+'.out')).open('x') as out:
                p=subprocess.run([sys.executable,*args],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            emit(event='exited',stage=label,code=p.returncode)
            if p.returncode:raise SystemExit(p.returncode)
        for label in ('baseline3k','oldgraph3k'):
            folder=WORK/label
            provenance=json.loads((folder/'provenance.json').read_text())
            assert hashlib.sha256((folder/'model.ckpt').read_bytes()).hexdigest()==provenance['sha256']
            cfg=OmegaConf.load(folder/'config.json')
            model=load_diffusion(cfg,str(folder),'model','cpu')
            emit(event='strict_load_passed',case=label,architecture=cfg.diffusion.denoiser._target_)
            del model
            run(label+'_generate',['-u','eval_part.py',f'run_dir={folder}','ckpt=model','input_type=val','num_runs=1','seed=9029','guidance=1.0','device=cuda','video_samples_per_run=0'])
            paths=list((folder/'generations/t2m').glob('val_model_*'));assert len(paths)==1
            for protocol in ('guo','guo+threshold'):
                run(label+'_'+protocol,['-u','-m','src.eval.run',str(paths[0]),'--dataset_name=frankenstein-dataset','--max_runs=1',f'--protocol={protocol}','--split=val','--eval_model_dir=pretrained/eval_model','--base_splits_root=datasets/annotations','--base_annotations_root=datasets/annotations','--device=cpu'])
            emit(event='case_complete',case=label,experiment_dir=str(paths[0]))
        emit(event='pipeline_complete')

if __name__=='__main__':main()
