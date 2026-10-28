"""Train the B2/B3 directional local-global fusion ablations from scratch.

The driver starts from the resolved no-dynamic-graph configuration so every
setting other than the requested attention direction matches the LG-FM trunk.
It deliberately requires a reference run containing frozen normalization
statistics; checkpoints and optimizer state are never reused.
"""
import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "experiments" / "lgfm_3k" / "no_graph_config.json"


def _replace_run_dir(value, old, new):
    if isinstance(value, dict):
        return {key: _replace_run_dir(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_run_dir(item, old, new) for item in value]
    return value.replace(old, new) if isinstance(value, str) else value


def prepare_config(direction, reference_run, output_run):
    config = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    old_run = config["run_dir"]
    config = _replace_run_dir(config, old_run, str(output_run))
    config.update(run_dir=str(output_run), resume_dir=None, init_ckpt=None, ckpt=None)
    denoiser = config["diffusion"]["denoiser"]
    denoiser.update(
        use_local_global_fusion=True,
        use_part_to_global_fusion=direction == "p2g",
        use_global_to_part_fusion=direction == "g2p",
        use_dynamic_kinematic_graph=False,
    )
    config["diffusion"].update(
        relation_affinity_loss_weight=0.0,
        relation_smoothness_loss_weight=0.0,
    )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("g2p", "p2g"), required=True)
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Existing LG-FM run that supplies only motion_stats and text_stats.",
    )
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()

    if args.output_run.exists():
        raise FileExistsError("Refusing to overwrite %s" % args.output_run)
    for name in ("motion_stats", "text_stats"):
        if not (args.reference_run / name).is_dir():
            raise FileNotFoundError(args.reference_run / name)

    config = prepare_config(args.direction, args.reference_run, args.output_run)
    assert config["dataloader"]["batch_size"] == 64
    assert config["trainer"]["max_epochs"] == 3000
    assert config["diffusion"]["denoiser"]["cross_attention_norm"] == "qk_g2p"

    args.output_run.mkdir(parents=True)
    for name in ("motion_stats", "text_stats"):
        shutil.copytree(args.reference_run / name, args.output_run / name)
    (args.output_run / "planned_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.output_run / "ablation_manifest.json").write_text(
        json.dumps(
            {
                "arm": "B2" if args.direction == "g2p" else "B3",
                "direction": args.direction,
                "reference_statistics": str(args.reference_run),
                "fresh_model_and_optimizer": True,
                "disabled": "part_to_global" if args.direction == "g2p" else "global_to_part",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from omegaconf import OmegaConf
    from train_part import train

    train.__wrapped__(OmegaConf.create(config))


if __name__ == "__main__":
    main()
