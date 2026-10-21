"""Matched epoch-499 generation: half refinement and no refinement."""
import json
import os
from pathlib import Path
import shutil
import torch
import scripts.evaluate_qk499 as evaluator

SOURCE=Path('outputs_amass/formal_qk_g2p_340_499_v1')
WORK=Path('logs/refine499_ablation_v1')

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    WORK.mkdir(exist_ok=False)
    for label in ('half','off'):
        staging=WORK/('source_'+label)
        (staging/'logs/checkpoints').mkdir(parents=True)
        cfg=json.loads((SOURCE/'config.json').read_text())
        ck=torch.load(SOURCE/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ck['epoch']==499
        gate=ck['state_dict']['denoiser.refine_gate']
        before=float(gate.sigmoid())
        if label=='half':
            ck['state_dict']['denoiser.refine_gate']=torch.logit(gate.sigmoid()*0.5)
        else:
            cfg['diffusion']['denoiser']['use_temporal_refine']=False
        for kind in ('motion','text'):
            name=kind+'_stats'
            shutil.copytree(SOURCE/name,staging/name)
            cfg['diffusion'][kind+'_normalizer']['base_dir']=str(staging/name)
        (staging/'config.json').write_text(json.dumps(cfg,indent=2))
        (staging/'intervention.json').write_text(json.dumps(dict(source=str(SOURCE),epoch=499,arm=label,original_gate=before,effective_gate=before*.5 if label=='half' else 0,seed=9029,full_reference='logs/formal_qk499_validation_v1',note='Inference-only intervention; no training updates.'),indent=2))
        torch.save(ck,staging/'logs/checkpoints/last.ckpt');del ck
        evaluator.SOURCE=staging
        evaluator.WORK=WORK/label
        print(json.dumps(dict(event='case_started',arm=label)),flush=True)
        evaluator.main()
        import scripts.audit_motion_dynamics as dynamics
        import scripts.audit_generated_jerk as jerk
        reference=Path('logs/formal_qk499_validation_v1/formal_g2p499_fullgraph/generations/t2m/val_model_20260929_140930')
        generated=list((evaluator.WORK/'formal_g2p499_fullgraph/generations/t2m').glob('val_model_*'))
        assert len(generated)==1
        dynamics.BASE=jerk.BASE=reference
        dynamics.NEW=jerk.NEW=generated[0]
        dynamics.main()
        shutil.copy2('logs/motion_dynamics_audit.json',evaluator.WORK/'dynamics_vs_full.json')
        jerk.main()
        shutil.copy2('logs/generated_jerk_audit.json',evaluator.WORK/'jerk_vs_full.json')
        print(json.dumps(dict(event='case_complete',arm=label)),flush=True)
    print('REFINE_ABLATION_COMPLETE',flush=True)

if __name__=='__main__':main()
