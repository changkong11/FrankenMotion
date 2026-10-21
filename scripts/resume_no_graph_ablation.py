"""Resume the unchanged ablation using data-disk temporary storage."""
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    os.chdir(Path(__file__).resolve().parents[1])
    target = Path('outputs_amass/ablation_no_dynamic_graph_qk_b64_3k_v1')
    temp = Path('/data_disk/wulw/FM/tmp/no_graph_ablation')
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    print('Waiting for GPU for ablation resume', flush=True)
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()
        candidates = [(int(r.split(',')[1]), r.split(',')[0].strip())
                      for r in rows if int(r.split(',')[1]) >= 36500]
        if candidates:
            break
        time.sleep(60)
    os.environ['CUDA_VISIBLE_DEVICES'] = max(candidates)[1]
    import tempfile
    tempfile.tempdir = str(temp)
    import torch
    torch.set_num_threads(4)
    from omegaconf import OmegaConf
    checkpoint = target / 'logs/checkpoints/last.ckpt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['epoch'] == 1182
    assert ck['optimizer_states']
    print(json.dumps(dict(event='resume_checkpoint_verified', epoch=ck['epoch'],
                          next_epoch=1183, global_step=ck['global_step'], temp_dir=str(temp))), flush=True)
    del ck
    cfg = json.loads((target / 'config.json').read_text())
    assert not cfg['diffusion']['denoiser']['use_dynamic_kinematic_graph']
    cfg.update(resume_dir=str(target), init_ckpt=None, ckpt=str(checkpoint))
    cfg['trainer']['max_epochs'] = 3000
    from train_part import train
    train.__wrapped__(OmegaConf.create(cfg))

if __name__ == '__main__':
    main()
