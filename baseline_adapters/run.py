"""CLI shared by MoScale and FloodDiffusion; native upstream entrypoints remain intact."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

from baseline_adapters.data import (MotionDataset, check_ready, collate, digest, dump,
                                    make_manifest, prepare_features)


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    for k in ('fm_root', 'repo_root', 'data_root', 'output_root'):
        cfg[k] = str(Path(os.path.expandvars(cfg[k])).resolve())
    protocol = check_ready(cfg['data_root'])
    protocol.update(conditioning=cfg['conditioning'], fps=20, max_frames=400, padded_frames=416)
    cfg['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return cfg


def initialize(cfg, stage, device, codec):
    sys.path.insert(0, cfg['repo_root'])
    from baseline_adapters.models import NativeBaseline
    return NativeBaseline(cfg, stage, device, codec)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(args, cfg):
    import torch
    from torch.utils.data import DataLoader
    seed_all(cfg['seed'])
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for production training')
    native = initialize(cfg, args.stage, args.device, args.codec)
    trainset = MotionDataset(cfg['data_root'], 'train', cfg['conditioning'])
    valset = MotionDataset(cfg['data_root'], 'val', cfg['conditioning'])
    if args.stage == 'generator':
        maximum = native.audit_text(trainset.rows + valset.rows)
        print(f'Text audit passed: maximum={maximum}, limit={native.max_tokens}', flush=True)
    settings = cfg['training'][args.stage]
    batch = settings['effective_batch_size']
    micro = args.micro_batch or settings['micro_batch_size']
    if batch % micro:
        raise ValueError('micro batch must divide effective batch')
    # Split each full effective batch explicitly, including the smaller last batch.
    loader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(valset, batch_size=micro, shuffle=False, num_workers=0, collate_fn=collate)
    out = Path(cfg['output_root']) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Existing run: use --resume or a new output_root')
    dump(out / 'resolved_config.json', cfg)
    audit = {'method': cfg['method'], 'stage': args.stage, 'protocol_hash': cfg['protocol_hash'],
        'codec_checkpoint': str(Path(args.codec).resolve()) if args.codec else None,
        'codec_sha256': digest(args.codec) if args.codec else None,
        'train_count': len(trainset), 'val_count': len(valset), 'selection': 'validation loss',
        'effective_batch': batch, 'micro_batch': micro, 'test_used_during_training': False,
        'upstream_files': {str(p.relative_to(cfg['repo_root'])): digest(p)
             for p in Path(cfg['repo_root']).rglob('*.py') if '.git' not in p.parts}}
    dump(out / 'provenance.json', audit)
    optimizer = torch.optim.AdamW([p for p in native.model.parameters() if p.requires_grad],
        lr=settings['lr'], betas=(0.9, 0.99), weight_decay=settings['weight_decay'])
    total_steps = settings.get('max_steps') or settings['epochs'] * len(loader)
    start_epoch, step, best, seconds = 0, 0, float('inf'), 0.
    # Track registered state only: deep-copying Flood's plain Python T5 wrapper copies an XXL encoder.
    ema = {k: v.detach().clone() for k,v in native.model.state_dict().items()} if settings.get('ema', 0) else None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['protocol_hash'] != cfg['protocol_hash'] or ckpt['stage'] != args.stage or ckpt['method'] != cfg['method']:
            raise ValueError('Resume protocol mismatch')
        if ckpt['codec_sha256'] != audit['codec_sha256']:
            raise ValueError('Resume codec differs')
        native.model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        native.restore_extra_state(ckpt.get('extra_state'))
        if ema is not None:
            ema = {k: v.to(native.device) for k,v in ckpt['ema'].items()}
        start_epoch, step, best, seconds = ckpt['epoch'], ckpt['step'], ckpt['best'], ckpt['seconds']
        random.setstate(ckpt['rng']['python']); np.random.set_state(ckpt['rng']['numpy'])
        torch.set_rng_state(ckpt['rng']['torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
    bf16 = args.device.startswith('cuda') and torch.cuda.is_bf16_supported()
    if args.device.startswith('cuda') and not bf16:
        print('BF16 unavailable: using FP32, recorded in training log', flush=True)
    start = time.monotonic()
    epoch = start_epoch
    while step < total_steps:
        native.model.train()
        if args.stage == 'generator':
            native.codec.eval()
        losses = []
        for rows, x, lengths in loader:
            warmup = min(1., (step+1) / settings['warmup_steps'])
            progress = max(0., (step-settings['warmup_steps']) / max(1, total_steps-settings['warmup_steps']))
            lr = settings['lr'] * warmup * (0.01 + 0.99 * (1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for offset in range(0, len(rows), micro):
                end = min(offset+micro, len(rows))
                with torch.autocast(device_type=native.device.type, dtype=torch.bfloat16, enabled=bf16):
                    loss = native.loss(rows[offset:end], x[offset:end], lengths[offset:end], True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}, keys={[r["key"] for r in rows[offset:end]]}')
                weight = (end-offset)/len(rows)
                (loss*weight).backward(); value += loss.item()*weight
            torch.nn.utils.clip_grad_norm_(native.model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; losses.append(value)
            if ema is not None:
                with torch.no_grad():
                    for key, q in native.model.state_dict().items():
                        if q.is_floating_point():
                            ema[key].lerp_(q, 1-settings['ema'])
                        else:
                            ema[key].copy_(q)
            if step % 20 == 0:
                print(f'epoch={epoch} step={step}/{total_steps} loss={value:.6f} lr={lr:.3g}', flush=True)
            if args.smoke_steps and step >= args.smoke_steps:
                # Smoke tests never produce production-ready checkpoints.
                dump(out / 'smoke.json', dict(steps=step, loss=value, status='train_steps_passed'))
                return
            if step >= total_steps:
                break
        epoch += 1
        # Validation does not affect optimization. Run it periodically to avoid repeatedly
        # encoding the same frozen T5 prompts while preserving resumable epoch checkpoints.
        validation_every = max(1, int(settings.get('validation_every_epochs', 1)))
        if epoch % validation_every != 0 and step < total_steps:
            elapsed = seconds + time.monotonic() - start
            record = dict(epoch=epoch, step=step, train_loss=float(np.mean(losses)), val_loss=None,
                          seconds=elapsed, precision='bf16' if bf16 else 'fp32', validation='skipped')
            state = dict(method=cfg['method'], stage=args.stage, protocol_hash=cfg['protocol_hash'],
                codec_sha256=audit['codec_sha256'], config=cfg, model=cpu_state(native.model),
                extra_state=native.extra_state(),
                ema={k:v.detach().cpu() for k,v in ema.items()} if ema is not None else None,
                optimizer=optimizer.state_dict(), epoch=epoch, step=step, best=best, seconds=elapsed,
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
            torch.save(state, out / 'last.tmp'); os.replace(out / 'last.tmp', out / 'last.pt')
            print(json.dumps(record), flush=True)
            continue
        # Fixed validation randomness; restore all training RNGs after validation.
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        seed_all(cfg['seed'] + 100000)
        raw = cpu_state(native.model) if ema is not None else None
        if ema is not None:
            native.model.load_state_dict(ema)
        native.model.eval(); val_sum, val_n = 0., 0
        with torch.no_grad():
            for rows, x, lengths in val_loader:
                with torch.autocast(device_type=native.device.type, dtype=torch.bfloat16, enabled=bf16):
                    loss = native.loss(rows, x, lengths, False)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                val_sum += float(loss)*len(rows); val_n += len(rows)
        if raw is not None:
            native.model.load_state_dict(raw)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
        if rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
        val = val_sum / val_n
        improved = val < best; best = min(best, val)
        record = dict(epoch=epoch, step=step, train_loss=float(np.mean(losses)), val_loss=val,
                      seconds=seconds+time.monotonic()-start, precision='bf16' if bf16 else 'fp32')
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        state = dict(method=cfg['method'], stage=args.stage, protocol_hash=cfg['protocol_hash'],
            codec_sha256=audit['codec_sha256'], config=cfg, model=cpu_state(native.model),
            extra_state=native.extra_state(),
            ema={k:v.detach().cpu() for k,v in ema.items()} if ema is not None else None, optimizer=optimizer.state_dict(),
            epoch=epoch, step=step, best=best, seconds=record['seconds'], rng=rng)
        torch.save(state, out / 'last.tmp'); os.replace(out / 'last.tmp', out / 'last.pt')
        if improved:
            shutil.copyfile(out / 'last.pt', out / 'best.pt')
        print(json.dumps(record), flush=True)


def generate(args, cfg):
    import torch
    from src.tools.guofeats import guofeats_to_joints
    seed_all(cfg['seed'])
    native = initialize(cfg, 'generator', args.device, args.codec)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['stage'] != 'generator' or ckpt['method'] != cfg['method'] or ckpt['protocol_hash'] != cfg['protocol_hash']:
        raise ValueError('Checkpoint provenance mismatch')
    if ckpt['codec_sha256'] != digest(args.codec):
        raise ValueError('Wrong codec for this generator')
    native.model.load_state_dict(ckpt.get('ema') or ckpt['model'], strict=True)
    dataset = MotionDataset(cfg['data_root'], args.split, cfg['conditioning'])
    native.audit_text(dataset.rows)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Generation directory must be new to avoid mixing runs')
    output.mkdir(parents=True)
    (output/'gt').mkdir()
    for row in dataset.rows:
        for ext in ('.npy', '.json'):
            shutil.copyfile(Path(cfg['data_root'])/'gt'/(row['key']+ext), output/'gt'/(row['key']+ext))
    provenance = dict(method=cfg['method'], conditioning=cfg['conditioning'], split=args.split,
                      checkpoint_sha256=digest(args.checkpoint), codec_sha256=digest(args.codec),
                      protocol_hash=cfg['protocol_hash'], seed=cfg['seed'], num_runs=args.runs,
                      total_samples=len(dataset), status='running', mesh_available=False)
    dump(output/'generation_manifest.json', provenance)
    with torch.no_grad():
        for run in range(args.runs):
            folder = output/f'turn_{run}'/'data'; folder.mkdir(parents=True)
            for row in dataset.rows:
                # Stable per-key seed: independent of batch size/order, with distinct run seeds.
                s = int(hashlib.sha256(f"{cfg['seed']}:{run}:{row['key']}".encode()).hexdigest()[:8],16)
                seed_all(s)
                with torch.autocast(device_type=native.device.type, dtype=torch.bfloat16,
                                    enabled=args.device.startswith('cuda') and torch.cuda.is_bf16_supported()):
                    feature = native.generate([row], torch.tensor([row['frames']]))[0]
                joints = guofeats_to_joints(feature.float()).numpy()
                if joints.shape != (row['frames'],22,3) or not np.isfinite(joints).all():
                    raise ValueError(f"Invalid output: {row['key']}")
                np.save(folder/(row['key']+'.npy'), {'joints':joints})
                shutil.copyfile(output/'gt'/(row['key']+'.json'), folder/(row['key']+'.json'))
            print(f'Generated run {run+1}/{args.runs}', flush=True)
    provenance['status'] = 'complete'
    dump(output/'generation_manifest.json', provenance)
    dump(output/'run_summary.json', provenance)


def evaluate(args, cfg):
    output = Path(args.output).resolve()
    summary = json.loads((output/'run_summary.json').read_text())
    if summary['status'] != 'complete' or summary['protocol_hash'] != cfg['protocol_hash']:
        raise ValueError('Incomplete generation or protocol mismatch')
    rows = json.loads((Path(cfg['data_root'])/'manifest.json').read_text(encoding='utf-8'))['splits'][summary['split']]
    expected = {r['key'] for r in rows}
    for folder in [output/'gt'] + [output/f'turn_{i}'/'data' for i in range(summary['num_runs'])]:
        if {p.stem for p in folder.glob('*.npy')} != expected or {p.stem for p in folder.glob('*.json')} != expected:
            raise ValueError(f'Incomplete coverage: {folder}')
    # Fail instead of allowing existing evaluator's graceful embedding-filter fallback.
    root = Path(cfg['fm_root'])/'datasets/annotations/frankenstein-dataset'
    if args.protocol == 'guo+threshold':
        for ext in ('.npy', '_index.json'):
            if not (root/'sent_embeddings/sentence-transformers'/('all-mpnet-base-v2'+ext)).exists():
                raise FileNotFoundError('Required threshold-filter embeddings unavailable')
    os.chdir(cfg['fm_root'])
    from src.eval.run import evaluate_experiment_enhanced
    evaluate_experiment_enhanced(str(output.resolve()), split_path=str(root), split=summary['split'],
        fps=20., max_runs=summary['num_runs'], protocol=args.protocol, batch_size=args.eval_batch,
        device=args.device, dataset_name='frankenstein-dataset',
        base_eval_model_dir=str(Path(cfg['fm_root'])/'pretrained/eval_model'),
        base_annotations_root=str(Path(cfg['fm_root'])/'datasets/annotations'), y_is_z_axis=False)


def physics(args, cfg):
    """Reuse the frozen supplemental metric definitions/reference without editing active jobs."""
    from scripts import evaluate_supplemental_physics as module
    experiment = Path(args.output).resolve()
    summary = json.loads((experiment/'run_summary.json').read_text())
    if summary['status'] != 'complete' or summary['protocol_hash'] != cfg['protocol_hash'] or summary['split'] != 'test':
        raise ValueError('Supplemental comparison requires complete matching test generation')
    output = experiment/'supplemental_physics'
    output.mkdir(exist_ok=True)
    reference_path = Path(args.reference).resolve()
    reference = json.loads(reference_path.read_text())
    annotations = json.loads((Path(cfg['fm_root'])/'datasets/annotations/frankenstein-dataset/annotations/annotations.json').read_text(encoding='utf-8'))
    dataset = MotionDataset(cfg['data_root'],'test',cfg['conditioning'])
    keys = sorted(r['key'] for r in dataset.rows)
    folders = [experiment/'gt'] + [experiment/f'turn_{i}'/'data' for i in range(summary['num_runs'])]
    for folder in folders:
        if sorted(p.stem for p in folder.glob('*.npy')) != keys:
            raise ValueError(f'Incomplete sample coverage: {folder}')
    manifest = dict(protocol_hash=cfg['protocol_hash'], reference_sha256=digest(reference_path),
        metric_sha256=digest(Path(cfg['fm_root'])/'src/eval/supplemental_metrics.py'),
        generation=summary, keys=keys)
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text()) != manifest:
        raise ValueError('Supplemental protocol changed: choose a new generation directory')
    dump(output/'manifest.json',manifest)
    dump(output/'training_reference.json',reference)
    module.evaluate_run(output,'GT',-1,experiment/'gt',keys,annotations,reference)
    for run in range(summary['num_runs']):
        module.evaluate_run(output,cfg['method'],run,experiment/f'turn_{run}'/'data',keys,annotations,reference)
    module.summarize(output,[cfg['method']],summary['num_runs'])
    dump(output/'complete.json',dict(samples=len(keys),runs=summary['num_runs']))


def preflight(args):
    import importlib.util
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    required=['torch','numpy','einops','omegaconf','transformers','flash_attn']
    if cfg['method']=='flooddiffusion': required += ['diffusers','ftfy']
    result={'dependencies':{m:importlib.util.find_spec(m) is not None for m in required},
            'data_ready':(Path(cfg['data_root'])/'ready.json').exists(),
            'features_converted':len(list((Path(cfg['data_root'])/'features').glob('*.npy')))}
    if cfg['method']=='flooddiffusion':
        result['text_weights']={k:Path(cfg['generator'][k]).exists() for k in ['checkpoint_path','tokenizer_path']}
    else:
        result['text_weights']='T5-base resolved by Hugging Face on generator initialization; cache/network needed'
    result['generator_environment_ready']=all(result['dependencies'].values()) and (
        all(result['text_weights'].values()) if isinstance(result['text_weights'],dict) else True)
    print(json.dumps(result,indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['manifest','prepare','audit','preflight','train','generate','evaluate','physics'])
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['codec','generator'], default='codec')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--codec'); p.add_argument('--checkpoint')
    p.add_argument('--resume'); p.add_argument('--output'); p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--runs', type=int, default=20); p.add_argument('--limit', type=int)
    p.add_argument('--micro-batch', type=int); p.add_argument('--smoke-steps', type=int)
    p.add_argument('--protocol', choices=['normal','guo+threshold'], default='guo+threshold')
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--reference', help='Frozen training_reference.json from the shared supplemental evaluation')
    args = p.parse_args()
    if args.command == 'preflight':
        result=preflight(args)
        if not result['generator_environment_ready']: sys.exit(2)
        return
    if args.command in ('manifest','prepare'):
        c = json.loads(Path(args.config).read_text(encoding='utf-8'))
        if args.command == 'manifest':
            m = make_manifest(c['fm_root'], c['data_root'])
            print({k:len(v) for k,v in m['splits'].items()})
        else:
            prepare_features(c['fm_root'], c['data_root'], args.limit)
        return
    cfg = resolve_config(args.config)
    if args.command == 'audit':
        print(json.dumps(check_ready(cfg['data_root']), indent=2)); return
    if args.command == 'train': train(args,cfg)
    elif args.command == 'generate':
        if not args.output or not args.checkpoint or not args.codec:
            p.error('generate needs --output, --checkpoint and --codec')
        generate(args,cfg)
    elif args.command == 'evaluate':
        if not args.output: p.error('evaluate needs --output')
        evaluate(args,cfg)
    elif args.command == 'physics':
        if not args.output or not args.reference: p.error('physics needs --output and --reference')
        physics(args,cfg)


if __name__ == '__main__':
    main()
