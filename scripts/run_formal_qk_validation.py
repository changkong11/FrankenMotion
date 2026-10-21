"""Formal recovery, then matched full-val generation/metrics in isolated folders."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.diagnose_graphlossfix import ROOT
from scripts.formal_qk_trial import TARGET

WORK=Path('logs/formal_qk_validation_v1')


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    WORK.mkdir(exist_ok=False)
    os.environ['FM_EVAL_TEXT_MODEL_PATH']=str(Path('pretrained/distilbert-base-uncased').resolve())
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**row):
            row['unix_time']=time.time()
            line=json.dumps(row);status.write(line+'\n');status.flush();print(line,flush=True)
        def run(label,args):
            emit(event='started',stage=label,command=args)
            with (WORK/f'{label}.out').open('x') as out:
                result=subprocess.run([sys.executable,*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            emit(event='exited',stage=label,returncode=result.returncode)
            if result.returncode:raise SystemExit(result.returncode)
        run('formal_train',['-u','-m','scripts.formal_qk_trial'])
        cases=[
            ('formal_g2p339_fullgraph',TARGET/'logs/checkpoints/last.ckpt','qk_g2p',0.5),
            ('original299_fullgraph',ROOT/'logs/checkpoints/recovery-epoch=299.ckpt','none',0.5),
            ('g2p339_bypass',Path('logs/controlled_qknorm_g2p_299_v1/graph_bypass_completed.ckpt'),'qk_g2p',0.0),
            ('bothqk339_bypass',Path('logs/controlled_qknorm_299_v1/graph_bypass_completed.ckpt'),'qk_both',0.0),
            ('qkv339_bypass',Path('logs/controlled_crossnorm_299_v1/graph_bypass_completed.ckpt'),'qkv_both',0.0),
        ]
        for label,ckpt,mode,cap in cases:
            folder=WORK/label;folder.mkdir()
            cfg=json.loads((ROOT/'config.json').read_text())
            cfg['run_dir']=str(folder)
            cfg['data']['preload']=False
            cfg['dataloader'].update(batch_size=8,num_workers=0)
            cfg['diffusion']['denoiser'].update(cross_attention_norm=mode,graph_residual_gain_cap=cap)
            for name in ('motion_stats','text_stats'):shutil.copytree(ROOT/name,folder/name)
            shutil.copy2(ckpt,folder/'model.ckpt')
            (folder/'config.json').write_text(json.dumps(cfg,indent=2))
            (folder/'comparison_manifest.json').write_text(json.dumps(dict(source_checkpoint=str(ckpt),normalization=mode,graph_cap=cap,split='val',seed=9029,num_runs=1,guidance=1.0,timesteps=cfg['diffusion']['timesteps'],note='One-seed full-val screening; different epochs/graph modes are not causal matched training comparisons.'),indent=2))
            run(label+'_generate',['-u','eval_part.py',f'run_dir={folder}','ckpt=model','input_type=val','num_runs=1','seed=9029','guidance=1.0','device=cuda','video_samples_per_run=0'])
            candidates=list((folder/'generations/t2m').glob('val_model_*'))
            assert len(candidates)==1
            exp=candidates[0]
            for protocol in ('guo','guo+threshold'):
                run(label+'_'+protocol,['-u','-m','src.eval.run',str(exp),'--dataset_name=frankenstein-dataset','--max_runs=1',f'--protocol={protocol}','--split=val','--eval_model_dir=pretrained/eval_model','--base_splits_root=datasets/annotations','--base_annotations_root=datasets/annotations','--device=cpu'])
            emit(event='case_complete',case=label,experiment_dir=str(exp))
        emit(event='validation_pipeline_complete')


if __name__=='__main__':main()
