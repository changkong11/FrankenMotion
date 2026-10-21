"""Matched test/20-run evaluation of the completed no-graph ablation."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.evaluate_qk2999_test20 import digest

WORK = Path('logs/no_graph_official_test20_v1')
SOURCE = Path('outputs_amass/ablation_no_dynamic_graph_qk_b64_3k_v1')
REFERENCE = Path('logs/qk2999_baseline_official_test20_v1')

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    WORK.mkdir(exist_ok=False)
    temp = Path('/data_disk/wulw/FM/tmp/no_graph_evaluation')
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
        FM_EVAL_TEXT_MODEL_PATH=str(Path('pretrained/distilbert-base-uncased').resolve()),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    torch.set_num_threads(4)
    from omegaconf import OmegaConf
    from src.tools.inference import load_diffusion
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**row):
            row['time']=time.time()
            line=json.dumps(row); status.write(line+'\n'); status.flush(); print(line,flush=True)
        def run(stage,args):
            emit(event='started',stage=stage,command=args)
            with (WORK/(stage+'.out')).open('x') as out:
                p=subprocess.run([sys.executable,'-u',*args],env=env,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            emit(event='exited',stage=stage,code=p.returncode)
            if p.returncode: raise RuntimeError(stage+' failed')
        manifest=json.loads((REFERENCE/'manifest.json').read_text())
        assert manifest['split']=='test' and manifest['num_runs']==20
        for path,checksum in manifest['evaluation_code'].items(): assert digest(Path(path))==checksum
        assert digest(Path('datasets/annotations/frankenstein-dataset/annotations/splits/test.txt'))==manifest['split_sha256']
        folder=WORK/'no_graph3k'; folder.mkdir()
        checkpoint=SOURCE/'logs/checkpoints/last.ckpt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['epoch']==2999
        assert not any('dynamic_graph_blocks' in k for k in ck['state_dict'])
        cfg=json.loads((SOURCE/'config.json').read_text())
        assert not cfg['diffusion']['denoiser']['use_dynamic_kinematic_graph']
        cfg['run_dir']=str(folder);cfg['data']['preload']=False
        cfg['dataloader'].update(batch_size=8,num_workers=0)
        for name in ('motion_stats','text_stats'):shutil.copytree(SOURCE/name,folder/name)
        shutil.copy2(checkpoint,folder/'model.ckpt')
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        model=load_diffusion(OmegaConf.create(cfg),str(folder),'model','cpu')
        del model,ck
        manifest['models']={'no_graph3k':dict(source=str(checkpoint),sha256=digest(checkpoint),epoch=2999)}
        (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2))
        emit(event='strict_load_passed',case='no_graph3k')
        emit(event='waiting_for_gpu')
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
            available=[(int(l.split(',')[1]),l.split(',')[0].strip()) for l in lines if int(l.split(',')[1])>=20000]
            if available:break
            time.sleep(60)
        env['CUDA_VISIBLE_DEVICES']=max(available)[1]
        run('generate',['eval_part.py',f'run_dir={folder}','ckpt=model','input_type=test',
            'num_runs=20','seed=9029','guidance=1.0','device=cuda','video_samples_per_run=0'])
        generated=list((folder/'generations/t2m').glob('test_model_*'));assert len(generated)==1
        exp=generated[0]
        keys=json.loads((REFERENCE/'sample_keys.json').read_text())
        for i in range(20):assert sorted(p.stem for p in (exp/f'turn_{i}/data').glob('*.npy'))==keys
        emit(event='generation_verified',runs=20,samples=len(keys))
        for protocol in ('guo','guo+threshold'):
            run(protocol,['-m','src.eval.run',str(exp),'--dataset_name=frankenstein-dataset',
                '--split=test','--max_runs=20',f'--protocol={protocol}','--device=cpu',
                '--eval_model_dir=pretrained/eval_model','--base_splits_root=datasets/annotations',
                '--base_annotations_root=datasets/annotations'])
        experiments=json.loads((REFERENCE/'experiments.json').read_text())
        experiments['no_graph3k']=str(exp)
        output=[]
        for protocol in ('guo','guo+threshold'):
            for subset in ('caption','action','parts_average'):
                cases={}
                for label,path in experiments.items():
                    with (Path(path)/f'evaluation_results_20_frankenstein-dataset_{protocol}_{subset}.csv').open() as f:rows=list(csv.DictReader(f))
                    assert len([r for r in rows if r['run'].startswith('run_')])==20
                    cases[label]={r['run']:r for r in rows}
                for metric in cases['no_graph3k']['AVERAGE']:
                    if metric=='run':continue
                    row=dict(protocol=protocol,subset=subset,metric=metric)
                    for label in cases:
                        for stat in ('AVERAGE','STD'):
                            value=float(cases[label][stat][metric]);assert math.isfinite(value)
                            row[label+'_'+stat.lower()]=value
                    output.append(row)
        with (WORK/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
        emit(event='pipeline_complete',comparison=str(WORK/'comparison.csv'))

if __name__=='__main__':main()
