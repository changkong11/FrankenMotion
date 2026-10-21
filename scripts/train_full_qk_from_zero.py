"""Full dynamic-graph model matched to the no-graph ablation from epoch zero."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path('outputs_amass/ablation_no_dynamic_graph_qk_b64_3k_v1')
OLD_FULL = Path('outputs_amass/formal_qk_g2p_2700_2999_v1')
TARGET = Path('outputs_amass/full_dynamic_graph_qk_from_zero_b64_3k_v1')

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    assert not TARGET.exists(), 'Refusing overwrite'
    reference = json.loads((SOURCE/'config.json').read_text())
    original = json.loads((OLD_FULL/'config.json').read_text())
    assert reference['resume_dir'] is None and reference['init_ckpt'] is None
    assert reference['diffusion']['denoiser']['cross_attention_norm']=='qk_g2p'
    assert not reference['diffusion']['denoiser']['use_dynamic_kinematic_graph']
    assert reference['dataloader']['batch_size']==64
    assert reference['trainer']['max_epochs']==3000
    assert reference['trainer']['precision']=='bf16-mixed'
    assert reference['diffusion']['lr']==1.675e-5
    cfg=json.loads(json.dumps(reference))
    changes={
        'diffusion.denoiser.use_dynamic_kinematic_graph':True,
        'diffusion.relation_affinity_loss_weight':original['diffusion']['relation_affinity_loss_weight'],
        'diffusion.relation_smoothness_loss_weight':original['diffusion']['relation_smoothness_loss_weight'],
    }
    for key,value in changes.items():
        node=cfg
        parts=key.split('.')
        for part in parts[:-1]:node=node[part]
        node[parts[-1]]=value
    assert changes['diffusion.relation_affinity_loss_weight']==0.02
    assert changes['diffusion.relation_smoothness_loss_weight']==0.002
    def compare(a,b,prefix=''):
        diffs=[]
        if isinstance(a,dict):
            assert set(a)==set(b)
            for key in a:diffs.extend(compare(a[key],b[key],prefix+'.'+key if prefix else key))
        elif isinstance(a,list):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):diffs.extend(compare(x,y,f'{prefix}.{i}'))
        elif a!=b:diffs.append(dict(key=prefix,no_graph=a,full=b))
        return diffs
    diffs=compare(reference,cfg)
    assert {d['key'] for d in diffs}==set(changes)
    def relocate(value):
        if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate(v) for v in value]
        return value.replace(str(SOURCE),str(TARGET)) if isinstance(value,str) else value
    cfg=relocate(cfg)
    assert cfg['ckpt'] is None and cfg['resume_dir'] is None and cfg['init_ckpt'] is None
    TARGET.mkdir()
    for name in ('motion_stats','text_stats'):shutil.copytree(SOURCE/name,TARGET/name)
    (TARGET/'planned_config.json').write_text(json.dumps(cfg,indent=2))
    (TARGET/'matched_ablation_manifest.json').write_text(json.dumps(dict(
        reference=str(SOURCE),reference_config_sha256=hashlib.sha256((SOURCE/'config.json').read_bytes()).hexdigest(),
        scientific_config_differences=diffs,seed=cfg['seed'],start_epoch=0,max_epochs=3000,
        note='Fresh parameters and optimizer; QK g2p enabled from epoch zero in both runs. Same seed does not ensure identical common-parameter initialization or RNG trajectory when architecture changes. Infrastructure interruption of no-graph reference cannot be exactly reproduced.'),indent=2))
    temp=Path('/data_disk/wulw/FM/tmp/full_qk_from_zero')
    temp.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    print(json.dumps(dict(event='config_verified',differences=diffs,seed=cfg['seed'])),flush=True)
    print('Waiting for GPU: minimum free 36500 MiB',flush=True)
    while True:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        candidates=[(int(l.split(',')[1]),l.split(',')[0].strip()) for l in lines if int(l.split(',')[1])>=36500]
        if candidates:break
        time.sleep(60)
    os.environ['CUDA_VISIBLE_DEVICES']=max(candidates)[1]
    import tempfile
    tempfile.tempdir=str(temp)
    import torch
    torch.set_num_threads(4)
    from omegaconf import OmegaConf
    from train_part import train
    train.__wrapped__(OmegaConf.create(cfg))
    ck=torch.load(TARGET/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==2999
    assert any('dynamic_graph_blocks' in key for key in ck['state_dict'])
    print('FULL_QK_FROM_ZERO_COMPLETE',flush=True)

if __name__=='__main__':main()
