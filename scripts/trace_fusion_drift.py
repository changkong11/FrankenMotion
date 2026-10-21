"""Instrument the established deterministic replay without changing updates."""
import json
import os
import sys
from pathlib import Path
import torch
from scripts.cross_attention_diagnostics import CrossAttentionDiagnostics
import scripts.reproduce_graphlossfix as replay

original_instantiate = replay.instantiate

def instrumented_instantiate(cfg, *args, **kwargs):
    model = original_instantiate(cfg, *args, **kwargs)
    if not hasattr(model, 'denoiser'):
        return model
    mode = os.environ.get('FM_TRACE_CROSS_NORM', 'none')
    if mode not in ('none', 'qk_g2p'):
        raise ValueError(mode)
    model.denoiser.cross_attention_norm = mode
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    metadata = dict(cross_attention_norm=mode, diagnostic_scope='actual attention inputs',
                    raw_local_scale='part_to_global_attn.raw_key.rms',
                    reference='logs/trace_fusion_drift_v1',
                    note='Both directions enabled; native g2p normalization consumes no random numbers.')
    (output/'trace_settings.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(event='trace_settings',**metadata)),flush=True)
    diag = CrossAttentionDiagnostics(model.denoiser, 'none')
    records = {}
    gradients = {}
    def capture(name, group):
        def hook(module, args, output):
            if not diag.enabled:
                return
            value = output[0] if isinstance(output, tuple) else output
            mask = diag.mask
            b,t = mask.shape
            p = len(model.denoiser.motion_part_names)
            if group == 'local_temporal_encoder':
                mask = mask[:,None].expand(b,p,t).reshape(b*p,t)
            elif group == 'part_interaction_encoder':
                mask = mask.reshape(b*t,1).expand(b*t,p)
            def rms(x):
                return float(x.detach().float()[mask].square().mean().sqrt())
            records[name] = rms(value)
            if value.requires_grad:
                value.register_hook(lambda g: gradients.update({name:rms(g)}))
        return hook
    for group in ('local_temporal_encoder','part_interaction_encoder','global_temporal_encoder'):
        for i,layer in enumerate(getattr(model.denoiser,group).layers):
            for branch in ('self_attn','linear2'):
                getattr(layer,branch).register_forward_hook(capture(f'{group}.{i}.{branch}',group))
    original_step = model.diffusion_step
    def step(batch, index, training=False, **kw):
        diag.enabled = index % 20 == 0
        if diag.enabled:
            records.clear(); gradients.clear(); diag.records.clear()
        result = original_step(batch,index,training=training,**kw)
        if diag.enabled:
            epoch=model._trainer.current_epoch
            print(json.dumps(dict(event='drift_forward',epoch=epoch,batch=index,
                                  branches=records,attention=diag.records)),flush=True)
            # Runs after backward, before clipping, through the wrapper below.
            model._drift_location=(epoch,index)
        else:
            model._drift_location=None
        return result
    model.diffusion_step=step
    original_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        if model._drift_location is not None:
            epoch,index=model._drift_location
            print(json.dumps(dict(event='drift_backward',epoch=epoch,batch=index,gradients=gradients)),flush=True)
        return original_clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=clip
    return model

if __name__=='__main__':
    replay.instantiate=instrumented_instantiate
    replay.main()
