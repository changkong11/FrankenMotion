"""Build upload manifests for the AMASS SMPL-RIFKE files used by this project."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "datasets" / "annotations" / "frankenstein-dataset" / "annotations"
MOTION_ROOT = ROOT / "datasets" / "motions" / "AMASS_20.0_fps_nh_smplrifke"
OUTPUT_ROOT = ROOT / "datasets" / "upload_manifests"


def paths_for_split(annotations: dict, split: str) -> list[str]:
    keyids = (ANNOTATION_ROOT / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    return sorted({f"{annotations[keyid]['path']}.npy" for keyid in keyids if keyid in annotations})


def write_manifest(name: str, paths: list[str]) -> None:
    missing = [path for path in paths if not (MOTION_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} references {len(missing)} missing files; first: {missing[0]}")

    output = OUTPUT_ROOT / f"amass_smplrifke_{name}.txt"
    output.write_text("\n".join(paths) + "\n", encoding="utf-8")
    total_bytes = sum((MOTION_ROOT / path).stat().st_size for path in paths)
    print(f"{name}: {len(paths)} files, {total_bytes / 1024**3:.3f} GB -> {output}")


def main() -> None:
    annotations = json.loads((ANNOTATION_ROOT / "annotations.json").read_text(encoding="utf-8"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_split = {split: paths_for_split(annotations, split) for split in ("train", "val", "test")}
    for split, paths in per_split.items():
        write_manifest(split, paths)

    write_manifest("train_val", sorted(set(per_split["train"]) | set(per_split["val"])))
    write_manifest("train_val_test", sorted(set().union(*per_split.values())))


if __name__ == "__main__":
    main()
