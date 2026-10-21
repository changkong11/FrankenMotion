"""Fresh no-dynamic-graph ablation using the repaired model configuration."""
import json, os, shutil, time, subprocess
from pathlib import Path
SOURCE=Path('outputs_amass/formal_qk_g2p_2700_2999_v1')
TARGET=Path('outputs_amass/ablation_no_dynamic_graph_qk_b64_3k_v1')

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    # Full generated-motion archival is optional; this ablation may start after
    # metrics are saved even when the large 20-run output transfer is deferred.
    assert not TARGET.exists(), 'Refusing overwrite'
    cfg=json.loads((SOURCE/'config.json').read_text())
    def relocate(v):
        if isinstance(v,dict): return {k:relocate(x) for k,x in v.items()}
        if isinstance(v,list): return [relocate(x) for x in v]
        return v.replace(str(SOURCE),str(TARGET)) if isinstance(v,str) else v
    cfg=relocate(cfg)
    cfg.update(resume_dir=None,init_ckpt=None,ckpt=None,run_dir=str(TARGET))
    cfg['diffusion']['denoiser']['use_dynamic_kinematic_graph']=False
    cfg['diffusion']['relation_affinity_loss_weight']=0.0
    cfg['diffusion']['relation_smoothness_loss_weight']=0.0
    cfg['trainer']['callbacks']=[c for c in cfg['trainer']['callbacks'] if not c['_target_'].startswith('scripts.')]
    assert cfg['diffusion']['denoiser']['cross_attention_norm']=='qk_g2p'
    assert cfg['dataloader']['batch_size']==64 and cfg['trainer']['max_epochs']==3000
    assert cfg['diffusion']['lr']==1.675e-5 and cfg['trainer']['precision']=='bf16-mixed'
    TARGET.mkdir()
    for name in ('motion_stats','text_stats'): shutil.copytree(SOURCE/name,TARGET/name)
    (TARGET/'planned_config.json').write_text(json.dumps(cfg,indent=2))
    (TARGET/'ablation_manifest.json').write_text(json.dumps(dict(reference=str(SOURCE),changes=['disable dynamic graph','zero relation affinity and smoothness losses','fresh model and optimizer'],history_note='QK g2p normalization active from epoch zero; reference introduced it at epoch 300. Same final configuration is not an identical training trajectory.'),indent=2))
    print('Waiting for GPU',flush=True)
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        candidates=[(int(r.split(',')[1]),r.split(',')[0].strip()) for r in rows if int(r.split(',')[1])>=36500]
        if candidates: break
        time.sleep(60)
    os.environ.update(CUDA_VISIBLE_DEVICES=max(candidates)[1],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4)
    from omegaconf import OmegaConf
    from train_part import train
    train.__wrapped__(OmegaConf.create(cfg))
    ck=torch.load(TARGET/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==2999
    assert not any('dynamic_graph_blocks' in k for k in ck['state_dict'])
    print('ABLATION_COMPLETE',flush=True)
if __name__=='__main__': main()
