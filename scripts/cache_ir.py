from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import COLAB_EXTRACT, output_dir
from har.constants import NUM_FRAMES
from har.loaders import load_frames
from har.paths import find_har_root, modality_trial_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=COLAB_EXTRACT)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    return p.parse_args()


def cache_ir(data_root: Path, index_csv: Path, force: bool = False, num_frames: int = NUM_FRAMES) -> Path:
    har_root = find_har_root(data_root)
    if har_root is None:
        raise SystemExit(f"No HAR tree under {data_root}")
    table = pd.read_csv(index_csv)
    ir_dir = index_csv.parent / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    n_ok = 0
    for i, row in table.iterrows():
        npy = ir_dir / f"{i:06d}.npy"
        if force or not npy.is_file():
            folder = modality_trial_dir(har_root, str(row["path_key"]), "ir")
            ir, ok = load_frames(folder, crop_person=True, num_frames=num_frames)
            if not ok:
                paths.append("")
                continue
            np.save(npy, ir.astype(np.float16))
        if npy.is_file():
            paths.append(str(npy))
            n_ok += 1
        else:
            paths.append("")
        if (i + 1) % 100 == 0:
            print(f"ir {i + 1}/{len(table)} ok={n_ok}")
    table["ir_npy"] = paths
    table.to_csv(index_csv, index=False)
    print(f"Wrote {index_csv} ir={n_ok}/{len(table)} num_frames={num_frames}")
    if n_ok:
        print(f"sample ir shape {np.load(next(p for p in paths if p)).shape}")
    return index_csv


def main() -> None:
    args = parse_args()
    index_csv = args.index or (output_dir("cache_v2") / "index.csv")
    cache_ir(args.data_root, index_csv, force=args.force, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
