"""Paired isolated replay from epoch 299; never writes to the source run."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import seed
from scripts.diagnose_graphlossfix import ROOT
from src.tools.inference import move_batch_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reset-adam', action='store_true')
    parser.add_argument('--normalize-cross-attention', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=['graph_on', 'graph_off', 'graph_bypass'], default=['graph_on', 'graph_off'])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / 'config.json')
    cfg.data.preload = False
    OmegaConf.save(cfg, output / 'source_config.yaml')
    (output / 'experiment.json').write_text(json.dumps(vars(args), indent=2))
    source = ROOT / 'logs/checkpoints/recovery-epoch=299.ckpt'
    for arm in args.arms:
        seed(9026)
        dataset = instantiate(cfg.data, split='train')
        loader = instantiate(cfg.dataloader, dataset=dataset, collate_fn=dataset.collate_fn,
                             shuffle=True, num_workers=0)
        model = instantiate(cfg.diffusion).cuda().train()
        ck = torch.load(source, map_location='cpu', weights_only=False)
        assert ck['epoch'] == 299
        model.load_state_dict(ck['state_dict'], strict=True)
        model._trainer = SimpleNamespace(current_epoch=300, max_epochs=3000)
        optimizer = model.configure_optimizers()['optimizer']
        optimizer.load_state_dict(ck['optimizer_states'][0])
        if args.reset_adam:
            optimizer.state.clear()
        del ck
        if arm in ('graph_off', 'graph_bypass'):
            model.relation_affinity_loss_weight = 0.0
            model.relation_smoothness_loss_weight = 0.0
        if arm == 'graph_bypass':
            assert not model.denoiser.use_contact_balance_state
            model.denoiser.graph_residual_gain_cap = 0.0
        if args.normalize_cross_attention:
            def normalize_inputs(module, inputs, kwargs):
                assert not inputs
                for key in ('query', 'key', 'value'):
                    value = kwargs[key]
                    kwargs[key] = torch.nn.functional.layer_norm(
                        value.float(), (value.shape[-1],), eps=1e-5
                    ).to(value.dtype)
                return inputs, kwargs
            for name in ('part_to_global_attn', 'global_to_part_attn'):
                getattr(model.denoiser, name).register_forward_pre_hook(
                    normalize_inputs, with_kwargs=True
                )
        model._trainer = SimpleNamespace(current_epoch=300, max_epochs=3000)
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        def norms(grads):
            result = {}
            for (name, _), grad in zip(params, grads):
                if grad is not None:
                    group = name.split('.')[1] if name.startswith('denoiser.') else name.split('.')[0]
                    result[group] = result.get(group, 0.0) + float(grad.detach().float().square().sum())
            return {k: v ** 0.5 for k, v in result.items()}
        with (output / f'{arm}.jsonl').open('x') as log:
            def emit(row):
                row['arm'] = arm
                line = json.dumps(row)
                log.write(line+'\n')
                log.flush()
                print(line, flush=True)
            emit(dict(event='start', source=str(source), normalize_cross_attention=args.normalize_cross_attention, reset_adam=args.reset_adam, lr=[g['lr'] for g in optimizer.param_groups], graph_residual_gain_cap=model.denoiser.graph_residual_gain_cap))
            stopped = False
            for epoch in range(300, 340):
                seed(100000 + epoch)
                model._trainer.current_epoch = epoch
                total = 0.0
                for index, cpu in enumerate(loader):
                    seed(epoch * 10000 + index)
                    batch = move_batch_to_device(copy.deepcopy(cpu), 'cuda')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        values = model.diffusion_step(batch, index, training=True)
                    loss = values['loss']
                    task = values.get('diff loss', loss)
                    if False:  # Component backward is deferred to saved-batch diagnostics.
                        components = {'task': task}
                        scale = min(1.0, (epoch+1)/model.coordination_loss_warmup_epochs)
                        for key, weight in [('relation affinity loss', model.relation_affinity_loss_weight),
                                            ('relation smoothness loss', model.relation_smoothness_loss_weight)]:
                            if key in values:
                                components[key] = values[key] * weight * scale
                        for key, component in components.items():
                            grads = torch.autograd.grad(component, [p for _, p in params], retain_graph=True, allow_unused=True)
                            emit(dict(event='component_gradient', epoch=epoch, batch=index, component=key, norms=norms(grads)))
                            del grads
                    loss.backward()
                    if index % 100 == 0:
                        emit(dict(event='total_gradient_modules', epoch=epoch, batch=index,
                                  norms=norms([p.grad for _, p in params])))
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False))
                    value = float(task.detach())
                    total += value
                    anomaly = not torch.isfinite(loss).item() or not torch.isfinite(torch.tensor(norm)).item() or norm > 100 or value > 0.12
                    if index % 20 == 0 or anomaly:
                        emit(dict(event='batch', epoch=epoch, batch=index, loss=value, grad_norm=norm,
                                  keyids=[str(k) for k in batch.get('keyid', [])]))
                    if anomaly:
                        torch.save(dict(state_dict=model.state_dict(), optimizer_states=[optimizer.state_dict()],
                                        epoch=epoch, batch=index, source=str(source), arm=arm, experiment=vars(args)), output / f'{arm}_anomaly.ckpt')
                        torch.save(cpu, output / f'{arm}_anomaly_batch.pt')
                        emit(dict(event='stopped_on_anomaly', epoch=epoch, batch=index))
                        stopped = True
                        break
                    optimizer.step()
                    del batch, values, loss, task
                emit(dict(event='epoch', epoch=epoch, mean_loss=total/(index+1)))
                if stopped:
                    break
            emit(dict(event='arm_finished', anomaly=stopped))
            if not stopped:
                torch.save(dict(state_dict=model.state_dict(), optimizer_states=[optimizer.state_dict()],
                                epoch=epoch, arm=arm, experiment=vars(args)), output / f'{arm}_completed.ckpt')
        del params, model, optimizer, loader, dataset
        torch.cuda.empty_cache()
    print('PAIRED_REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
