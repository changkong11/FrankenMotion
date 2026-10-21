"""Continue validated full-graph run for epochs 340--499 in an isolated folder."""
import json
import logging
import os
import shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from scripts.formal_qk_trial import FormalAudit
import scripts.formal_qk_trial as original
from scripts.cross_attention_diagnostics import CrossAttentionDiagnostics

SOURCE=Path(os.environ.get('FM_EXTEND_SOURCE','outputs_amass/formal_qk_g2p_299_339_v1'))
TARGET=Path(os.environ.get('FM_EXTEND_TARGET','outputs_amass/formal_qk_g2p_340_499_v1'))
START=int(os.environ.get('FM_EXTEND_START','340'))
END=int(os.environ.get('FM_EXTEND_END','499'))

class ExtendedAudit(FormalAudit):
    def on_train_start(self,trainer,model):
        original.TARGET=TARGET
        assert trainer.current_epoch==START
        assert model.denoiser.cross_attention_norm=='qk_g2p'
        assert model.denoiser.graph_residual_gain_cap==0.5
        assert model.relation_affinity_loss_weight==0.02
        assert model.relation_smoothness_loss_weight==0.002
        lrs=[g['lr'] for g in trainer.optimizers[0].param_groups]
        assert len(lrs)==2 and all(abs(a-b)<1e-12 for a,b in zip(lrs,[1.675e-5,1.675e-6]))
        steps=[float(v['step']) for v in trainer.optimizers[0].state.values() if 'step' in v]
        assert steps and min(steps)>0
        self.diag=CrossAttentionDiagnostics(model.denoiser,'none')
        print(json.dumps(dict(event='extended_start',epoch=trainer.current_epoch,lr=lrs,adam_steps=[min(steps),max(steps)],graph_cap=0.5)),flush=True)

    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        self.diag.enabled=batch_idx%40==0

    def on_before_optimizer_step(self,trainer,model,optimizer):
        if self.diag.enabled:
            norm=sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None)**0.5
            print(json.dumps(dict(event='extended_diagnostics',epoch=trainer.current_epoch,step=trainer.global_step,gradient_norm=norm,attention=self.diag.records)),flush=True)
            if not torch.isfinite(torch.tensor(norm)) or norm>100:
                trainer.save_checkpoint(str(TARGET/'logs/checkpoints/gradient_stop.ckpt'))
                raise RuntimeError('Gradient stop before update')

def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    if TARGET.exists():raise RuntimeError(f'Refusing overwrite: {TARGET}')
    cfg=json.loads((SOURCE/'config.json').read_text())
    def relocate(x):
        if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
        if isinstance(x,list):return [relocate(v) for v in x]
        return x.replace(str(SOURCE),str(TARGET)) if isinstance(x,str) else x
    cfg=relocate(cfg)
    cfg['trainer']['max_epochs']=END+1
    cfg['trainer']['callbacks']=[c for c in cfg['trainer']['callbacks'] if c.get('_target_') not in ('scripts.formal_qk_trial.FormalAudit','scripts.extend_formal_qk.ExtendedAudit')]
    cfg['trainer']['callbacks'].append({'_target_':'scripts.extend_formal_qk.ExtendedAudit'})
    cfg['resume_dir']=str(TARGET);cfg['init_ckpt']=None
    cfg['ckpt']=str(TARGET/'recovery_input.ckpt')
    TARGET.mkdir(parents=True)
    for name in ('motion_stats','text_stats'):shutil.copytree(SOURCE/name,TARGET/name)
    (TARGET/'config.json').write_text(json.dumps(cfg,indent=2))
    ck=torch.load(SOURCE/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==START-1
    ck['callbacks']={}
    torch.save(ck,cfg['ckpt']);del ck
    from train_part import train
    train.__wrapped__(OmegaConf.create(cfg))
    ck=torch.load(TARGET/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==END
    print('EXTENDED_FORMAL_QK_COMPLETE',flush=True)

if __name__=='__main__':main()
