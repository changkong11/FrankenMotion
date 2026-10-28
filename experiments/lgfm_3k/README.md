# LG-FM 3000-epoch experiments

`no_graph_config.json` and `full_config.json` are the resolved configurations of
the completed batch-64 runs. Both use fresh model/optimizer initialization,
BF16, seed 1234 and Q/K normalization from epoch zero. The full model enables
the dynamic graph and its relation losses. Do not substitute the repository
defaults for these resolved configurations when reproducing these experiments.

Install the project environment (`environment_fixed.yml` or
`requirements_fixed.txt`) and prepare the licensed datasets and pretrained text
and evaluation encoders separately. Checkpoints, datasets, normalization
statistics and generated motions are intentionally not included in Git.

Run commands from the repository root. The existing experiment drivers preserve
the original server paths and refuse to overwrite existing run directories:

- `python -m scripts.train_no_graph_ablation`
- `python -m scripts.train_full_qk_from_zero`
- `python -m scripts.train_directional_fusion_ablation --direction g2p --reference-run <trunk_run> --output-run <b2_run>`
- `python -m scripts.train_directional_fusion_ablation --direction p2g --reference-run <trunk_run> --output-run <b3_run>`
- `python -m scripts.evaluate_qk2999_test20`
- `python -m scripts.evaluate_no_graph_test20`
- `python -m scripts.evaluate_full_from_zero_test20`

These drivers require historical `outputs_amass` reference configurations and
motion/text statistics. On a new machine, restore those artifacts or adapt the
source/target paths using the configuration snapshots here. They are archival
experiment drivers, not self-contained dataset download tools. For direct fresh
training, load a snapshot with OmegaConf, relocate `run_dir` and normalizer paths,
clear `resume_dir`, `init_ckpt`, and `ckpt`, then call
`train_part.train.__wrapped__(cfg)` as the drivers do.

Evaluation uses the test split, 20 runs, seed 9029 with stride 1000,
guidance 1.0 and generation batch size 8. Both `guo` and `guo+threshold`
protocols are computed on caption, action and parts-average subsets. Evaluation
drivers verify historical evaluator and split hashes and require the preceding
reference manifests/results to construct comparisons. New kinematic Gap metrics
and the matched 5-second FID+ crop implementation are in `src/eval/run.py`.

`scripts/audit_*`, `scripts/reproduce_*`, `scripts/trace_*` and the Q/K tests
preserve the numerical-instability investigation. Most require checkpoints and
data; they are not lightweight unit tests.

## Directional fusion ablations (B2/B3)

`use_part_to_global_fusion` and `use_global_to_part_fusion` independently
enable the two cross-attention paths.  Existing configurations that omit these
keys keep their former bidirectional behavior whenever
`use_local_global_fusion=true`.  The B2 command enables only global-to-part;
the B3 command enables only part-to-global.  Each driver begins with a fresh
model and optimizer, copies only normalization statistics from
`--reference-run`, and refuses to overwrite `--output-run`.
