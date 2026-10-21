"""Matched test/20-run evaluation of repaired 3K and baseline 3K checkpoints."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path('logs/qk2999_baseline_official_test20_v1')
SOURCES = {
    'qk2999': Path('logs/formal_qk2999_validation_v1/formal_g2p2999_fullgraph'),
    'baseline3k': Path('logs/matched_historical_v1/baseline3k'),
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    WORK.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(FM_EVAL_TEXT_MODEL_PATH=str(Path('pretrained/distilbert-base-uncased').resolve()),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    from omegaconf import OmegaConf
    from src.tools.inference import load_diffusion
    torch.set_num_threads(4)
    with (WORK/'status.jsonl').open('x') as status:
        def emit(**row):
            row['time'] = time.time()
            line = json.dumps(row)
            status.write(line+'\n'); status.flush(); print(line, flush=True)
        def run(label, args):
            emit(event='started', stage=label, command=args)
            with (WORK/(label+'.out')).open('x') as out:
                result = subprocess.run([sys.executable, '-u', *args], env=env,
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            emit(event='exited', stage=label, code=result.returncode)
            if result.returncode:
                raise RuntimeError(label+' failed')
        annotation_root = Path('datasets/annotations/frankenstein-dataset')
        for name in ('all-mpnet-base-v2.npy', 'all-mpnet-base-v2_index.json'):
            assert (annotation_root/'sent_embeddings/sentence-transformers'/name).is_file()
        manifest = dict(split='test', num_runs=20, seed=9029, seed_stride=1000,
            guidance=1.0, generation_batch_size=8, metric_device='cpu',
            protocols=['guo','guo+threshold'], subsets=['caption','action','parts_average'],
            split_sha256=digest(annotation_root/'annotations/splits/test.txt'), models={},
            evaluation_code={str(p):digest(p) for p in [Path('eval_part.py'),Path('src/eval/run.py'),Path('src/eval/load_generated_data.py')]},
            note='Project evaluation code, including existing metric fixes; detailed metrics preserve raw column names. Legacy paper_table FID_nrom column is absent in this evaluator; do not relabel raw FID as normalized FID.')
        for label, source in SOURCES.items():
            folder = WORK/label
            folder.mkdir()
            checkpoint = source/'model.ckpt'
            checksum = digest(checkpoint)
            if label=='baseline3k':
                assert checksum==json.loads((source/'provenance.json').read_text())['sha256']
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert ck['epoch']==2999
            cfg = json.loads((source/'config.json').read_text())
            cfg['run_dir']=str(folder)
            cfg['data']['preload']=False
            cfg['dataloader'].update(batch_size=8,num_workers=0)
            if label=='qk2999':
                assert cfg['diffusion']['denoiser']['cross_attention_norm']=='qk_g2p'
            for stats in ('motion_stats','text_stats'):
                shutil.copytree(source/stats, folder/stats)
            shutil.copy2(checkpoint, folder/'model.ckpt')
            (folder/'config.json').write_text(json.dumps(cfg,indent=2))
            model = load_diffusion(OmegaConf.create(cfg),str(folder),'model','cpu')
            manifest['models'][label]=dict(source=str(checkpoint),sha256=checksum,epoch=ck['epoch'],global_step=ck['global_step'])
            del model, ck
            emit(event='strict_load_passed',case=label,sha256=checksum)
        (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2))
        experiments={}
        for label in SOURCES:
            emit(event='waiting_for_gpu',case=label)
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
                candidates=[(int(r.split(',')[1]),r.split(',')[0].strip()) for r in rows if int(r.split(',')[1])>=20000]
                if candidates: break
                time.sleep(60)
            env['CUDA_VISIBLE_DEVICES']=max(candidates)[1]
            folder=WORK/label
            run(label+'_generate',['eval_part.py',f'run_dir={folder}','ckpt=model','input_type=test','num_runs=20','seed=9029','guidance=1.0','device=cuda','video_samples_per_run=0'])
            generated=list((folder/'generations/t2m').glob('test_model_*'))
            assert len(generated)==1
            exp=generated[0]; experiments[label]=str(exp)
            reference_keys=None
            for i in range(20):
                keys=sorted(p.stem for p in (exp/f'turn_{i}/data').glob('*.npy'))
                assert keys, (label,i)
                if reference_keys is None: reference_keys=keys
                assert keys==reference_keys, (label,i,'sample mismatch')
            keypath=WORK/'sample_keys.json'
            if keypath.exists(): assert reference_keys==json.loads(keypath.read_text())
            else: keypath.write_text(json.dumps(reference_keys))
            emit(event='generation_verified',case=label,runs=20,samples=len(reference_keys))
            for protocol in ('guo','guo+threshold'):
                run(label+'_'+protocol,['-m','src.eval.run',str(exp),'--dataset_name=frankenstein-dataset','--split=test','--max_runs=20',f'--protocol={protocol}','--device=cpu','--eval_model_dir=pretrained/eval_model','--base_splits_root=datasets/annotations','--base_annotations_root=datasets/annotations'])
            emit(event='case_complete',case=label,experiment_dir=str(exp))
        comparison=[]
        for protocol in ('guo','guo+threshold'):
            for subset in ('caption','action','parts_average'):
                cases={}
                for label,exp in experiments.items():
                    path=Path(exp)/f'evaluation_results_20_frankenstein-dataset_{protocol}_{subset}.csv'
                    with path.open() as f: rows=list(csv.DictReader(f))
                    assert len([r for r in rows if r['run'].startswith('run_')])==20
                    cases[label]={r['run']:r for r in rows}
                for metric in cases['qk2999']['AVERAGE']:
                    if metric=='run': continue
                    row=dict(protocol=protocol,subset=subset,metric=metric)
                    for label in SOURCES:
                        for stat in ('AVERAGE','STD'):
                            value=float(cases[label][stat][metric]); assert math.isfinite(value)
                            row[label+'_'+stat.lower()]=value
                    comparison.append(row)
        with (WORK/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
        (WORK/'experiments.json').write_text(json.dumps(experiments,indent=2))
        emit(event='pipeline_complete',comparison=str(WORK/'comparison.csv'))

if __name__=='__main__':
    main()
