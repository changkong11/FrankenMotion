"""Full validation-set screening matched to the existing epoch-339 evaluation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

SOURCE=Path(os.environ.get('FM_EVAL_SOURCE','outputs_amass/formal_qk_g2p_340_499_v1'))
WORK=Path(os.environ.get('FM_EVAL_WORK','logs/formal_qk499_validation_v1'))
EPOCH=int(os.environ.get('FM_EVAL_EPOCH','499'))

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    checkpoint=SOURCE/'logs/checkpoints/last.ckpt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==EPOCH
    del ck
    WORK.mkdir(exist_ok=False)
    os.environ.update(FM_EVAL_TEXT_MODEL_PATH=str(Path('pretrained/distilbert-base-uncased').resolve()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**row):
            row['unix_time']=time.time()
            line=json.dumps(row); status.write(line+'\n'); status.flush(); print(line,flush=True)
        def run(label,args):
            emit(event='started',stage=label,command=args)
            with (WORK/(label+'.out')).open('x') as out:
                result=subprocess.run([sys.executable,*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            emit(event='exited',stage=label,returncode=result.returncode)
            if result.returncode:raise SystemExit(result.returncode)
        folder=WORK/f'formal_g2p{EPOCH}_fullgraph';folder.mkdir()
        cfg=json.loads((SOURCE/'config.json').read_text())
        cfg['run_dir']=str(folder)
        cfg['data']['preload']=False
        cfg['dataloader'].update(batch_size=8,num_workers=0)
        assert cfg['diffusion']['denoiser']['cross_attention_norm']=='qk_g2p'
        assert cfg['diffusion']['denoiser']['graph_residual_gain_cap']==0.5
        for name in ('motion_stats','text_stats'):shutil.copytree(SOURCE/name,folder/name)
        shutil.copy2(checkpoint,folder/'model.ckpt')
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        (folder/'comparison_manifest.json').write_text(json.dumps(dict(source_checkpoint=str(checkpoint),epoch=EPOCH,split='val',seed=9029,num_runs=1,guidance=1.0,reference='logs/formal_qk_validation_v1/formal_g2p339_fullgraph',note='Single-generation-run screening, not a significance test.'),indent=2))
        run('generate',['-u','eval_part.py',f'run_dir={folder}','ckpt=model','input_type=val','num_runs=1','seed=9029','guidance=1.0','device=cuda','video_samples_per_run=0'])
        candidates=list((folder/'generations/t2m').glob('val_model_*'))
        assert len(candidates)==1
        exp=candidates[0]
        for protocol in ('guo','guo+threshold'):
            run(protocol,['-u','-m','src.eval.run',str(exp),'--dataset_name=frankenstein-dataset','--max_runs=1',f'--protocol={protocol}','--split=val','--eval_model_dir=pretrained/eval_model','--base_splits_root=datasets/annotations','--base_annotations_root=datasets/annotations','--device=cpu'])
        emit(event='validation_pipeline_complete',experiment_dir=str(exp))

if __name__=='__main__':main()
