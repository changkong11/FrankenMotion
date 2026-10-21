"""Read-only fixed-input checkpoint and module substitution probe."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import seed
from src.tools.inference import move_batch_to_device

ROOT = Path('outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset_lgfm_scratch_bf16_b64_graphlossfix_lr1675_3k')


def main():
    torch.set_num_threads(4)
    checkpoints = {}
    for epoch in (299, 349, 399):
        ck = torch.load(ROOT / f'logs/checkpoints/recovery-epoch={epoch}.ckpt', map_location='cpu', weights_only=False)
        checkpoints[epoch] = ck['state_dict']
        opt = ck['optimizer_states'][0]
        steps = [float(s['step']) for s in opt['state'].values() if 'step' in s]
        print(json.dumps(dict(epoch=epoch, optimizer_steps=[min(steps), max(steps)], lr=[g['lr'] for g in opt['param_groups']], finite=all(bool(torch.isfinite(t).all()) for t in ck['state_dict'].values() if t.is_floating_point()))), flush=True)
        del ck, opt
    same = all(torch.equal(checkpoints[349][k], checkpoints[399][k]) for k in checkpoints[349])
    print(json.dumps(dict(weights_349_equal_399=same)), flush=True)
    seed(9025)
    cfg = OmegaConf.load(ROOT / 'config.json')
    cfg.data.preload = False
    data = instantiate(cfg.data, split='train')
    batches = [data.collate_fn([data[i] for i in range(start, start+4)]) for start in (0, 64)]
    model = instantiate(cfg.diffusion).cuda().eval()
    model._trainer = SimpleNamespace(current_epoch=349, max_epochs=3000)
    groups = {
        'graph': ('dynamic_graph_blocks.',),
        'global': ('global_temporal_encoder.',),
        'local': ('local_temporal_encoder.', 'part_interaction_encoder.'),
        'fusion': ('part_to_global_attn.', 'global_to_part_attn.', 'coord_projection.', 'coord_gate'),
        'heads': ('base_motion_head.', 'text_head.', 'refine_tcn.', 'refine_motion_head.', 'refine_gate'),
        'embedding': ('tx_embedding.', 'motion_global_embedding.', 'action_embedding.', 'part_text_embedding.', 'motion_part_embeddings.', 'part_id_embedding.', 'timestep_encoder.'),
    }
    good, bad = checkpoints[299], checkpoints[349]
    for label in ('good', 'bad', *groups):
        state = dict(good if label == 'good' else bad)
        replaced = []
        if label in groups:
            for key in state:
                if any(key.startswith('denoiser.'+p) for p in groups[label]):
                    state[key] = good[key]
                    replaced.append(key)
        model.load_state_dict(state, strict=True)
        for precision in ('fp32', 'bf16') if label in ('good', 'bad') else ('bf16',):
            losses = []
            for i, cpu in enumerate(batches):
                seed(1200+i)
                batch = move_batch_to_device(copy.deepcopy(cpu), 'cuda')
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                    values = model.diffusion_step(batch, i, training=False)
                losses.append(float(values['diff loss']))
            print(json.dumps(dict(replaced=label, tensors=len(replaced), precision=precision, losses=losses)), flush=True)


if __name__ == '__main__':
    main()
