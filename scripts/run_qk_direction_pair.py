"""Run the two isolated directional ablations sequentially on one GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    status=root/'logs/qk_direction_pair_v1.status.jsonl'
    jobs=[('global_to_part_attn','controlled_qknorm_g2p_299_v1'),
          ('part_to_global_attn','controlled_qknorm_p2g_299_v1')]
    for _,name in jobs:
        if (root/'logs'/name).exists() or (root/'logs'/f'{name}.out').exists():
            raise RuntimeError(f'Refusing to overwrite {name}')
    with status.open('x') as progress:
        def emit(**row):
            row['unix_time']=time.time()
            line=json.dumps(row)
            progress.write(line+'\n');progress.flush();print(line,flush=True)
        for direction,name in jobs:
            cmd=[sys.executable,'-u','-m','scripts.reproduce_qk_direction',
                 '--output',f'logs/{name}','--normalize-cross-attention',
                 '--arms','graph_bypass','--direction',direction,'--end-epoch','339']
            with (root/'logs'/f'{name}.out').open('x') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                emit(event='started',direction=direction,pid=child.pid,command=cmd)
                code=child.wait()
            emit(event='exited',direction=direction,exit_code=code)
            if code:
                raise SystemExit(code)
        emit(event='pair_complete')


if __name__=='__main__':
    main()
