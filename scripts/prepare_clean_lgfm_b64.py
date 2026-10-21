"""Freeze the historical batch-48 config with batch size 64 and a verified trunk."""
import copy
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset_coord_graph_gated_init_3k/evaluation/guo_threshold_20/config.json'
TRUNK = ROOT / 'outputs_amass_LG-FM/frankenmotion_clip_part_smplrifke_frankenstein-dataset/logs/checkpoints/lgfm_3k.ckpt'

def main():
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    checkpoint = torch.load(TRUNK, map_location='cpu', weights_only=False)
    graph_keys = [k for k in checkpoint['state_dict'] if 'dynamic_graph_blocks' in k]
    assert checkpoint['epoch'] == 2999 and not graph_keys
    config = copy.deepcopy(original)
    config['dataloader']['batch_size'] = 64
    config['init_ckpt'] = TRUNK.relative_to(ROOT).as_posix()
    suffix = '_coord_graph_gated_clean3k_b64'
    run_dir = 'outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset' + suffix
    config['run_suffix'] = suffix
    config['run_dir'] = run_dir
    for kind in ('motion', 'text'):
        config['diffusion'][kind + '_normalizer']['base_dir'] = run_dir + '/' + kind + '_stats'
    for logger in config['trainer']['logger']:
        logger['save_dir'] = run_dir
    dest = ROOT / 'logging/clean_lgfm_b64'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    report = dict(checkpoint=str(TRUNK), sha256=hashlib.sha256(TRUNK.read_bytes()).hexdigest(),
                  epoch=checkpoint['epoch'], global_step=checkpoint['global_step'], graph_key_count=len(graph_keys),
                  training_change={'dataloader.batch_size': [48,64]},
                  other_changes='Verified source checkpoint path and isolated output paths only. Current code defaults and runtime behavior still need preflight verification before launch.')
    (dest / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
