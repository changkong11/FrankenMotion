"""Isolated run through train_part.py's actual Lightning training entry point."""
import json
import logging
import shutil
from pathlib import Path
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from scripts.diagnose_graphlossfix import ROOT

TARGET=Path('outputs_amass/formal_qk_g2p_299_339_v1')


class FormalAudit(pl.Callback):
    def on_train_start(self,trainer,model):
        assert trainer.current_epoch==300
        assert model.denoiser.cross_attention_norm=='qk_g2p'
        assert model.denoiser.graph_residual_gain_cap==0.5
        assert not model.denoiser.global_to_part_attn._forward_pre_hooks
        lrs=[g['lr'] for g in trainer.optimizers[0].param_groups]
        assert all(abs(a-b)<1e-12 for a,b in zip(lrs,[1.675e-5,1.675e-6]))
        steps=[float(v['step']) for v in trainer.optimizers[0].state.values() if 'step'in v]
        assert steps and min(steps)>0
        print(json.dumps(dict(event='formal_start',epoch=trainer.current_epoch,global_step=trainer.global_step,lr=lrs,adam_steps=[min(steps),max(steps)])),flush=True)

    def on_before_backward(self,trainer,model,loss):
        if not torch.isfinite(loss).item() or float(loss.detach())>0.12:
            trainer.save_checkpoint(str(TARGET/'logs/checkpoints/absolute_loss_stop.ckpt'))
            raise RuntimeError('Formal diagnostic absolute loss stop before update')

    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if batch_idx%40==0:
            loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
            print(json.dumps(dict(event='formal_batch',epoch=trainer.current_epoch,batch=batch_idx,global_step=trainer.global_step,loss=None if loss is None else float(loss.detach()))),flush=True)

    def on_validation_epoch_end(self,trainer,model):
        print(json.dumps(dict(event='formal_validation',epoch=trainer.current_epoch,metrics={k:float(v) for k,v in trainer.callback_metrics.items() if k.startswith('val_') and torch.is_tensor(v) and v.numel()==1})),flush=True)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    if TARGET.exists():raise RuntimeError(f'Refusing overwrite: {TARGET}')
    cfg=json.loads((ROOT/'config.json').read_text())
    def relocate(x):
        if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
        if isinstance(x,list):return [relocate(v) for v in x]
        return x.replace(str(ROOT),str(TARGET)) if isinstance(x,str) else x
    cfg=relocate(cfg)
    cfg['diffusion']['denoiser']['cross_attention_norm']='qk_g2p'
    cfg['trainer']['max_epochs']=340
    cfg['trainer']['callbacks'].append({'_target_':'scripts.formal_qk_trial.FormalAudit'})
    cfg['data']['preload']=False
    cfg['resume_dir']=str(TARGET)
    cfg['init_ckpt']=None
    cfg['ckpt']=str(TARGET/'recovery_input.ckpt')
    TARGET.mkdir(parents=True)
    for name in ('motion_stats','text_stats'):shutil.copytree(ROOT/name,TARGET/name)
    (TARGET/'config.json').write_text(json.dumps(cfg,indent=2))
    ck=torch.load(ROOT/'logs/checkpoints/recovery-epoch=299.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==299
    ck['callbacks']={}  # Prevent source checkpoint callbacks deleting source files.
    ck['formal_qk_trial']={'normalization':'qk_g2p','source':str(ROOT),'source_epoch':299}
    torch.save(ck,cfg['ckpt']);del ck
    from train_part import train
    train.__wrapped__(OmegaConf.create(cfg))
    final=TARGET/'logs/checkpoints/last.ckpt'
    ck=torch.load(final,map_location='cpu',weights_only=False)
    assert ck['epoch']==339, f'Incomplete formal trial: {ck["epoch"]}'
    print('FORMAL_QK_TRIAL_COMPLETE',flush=True)


if __name__=='__main__':main()
