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
from har.loaders import load_skeleton
from har.paths import find_har_root, modality_trial_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=COLAB_EXTRACT)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def cache_skeletons(data_root: Path, index_csv: Path, force: bool = False) -> Path:
    har_root = find_har_root(data_root)
    if har_root is None:
        raise SystemExit(f"No HAR tree under {data_root}")
    table = pd.read_csv(index_csv)
    skel_dir = index_csv.parent / "skeleton"
    skel_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    n_ok = 0
    for i, row in table.iterrows():
        npy = skel_dir / f"{i:06d}.npy"
        if force or not npy.is_file():
            folder = modality_trial_dir(har_root, str(row["path_key"]), "skeleton")
            skel, ok = load_skeleton(folder)
            if not ok:
                paths.append("")
                continue
            np.save(npy, skel.astype(np.float16))
        if npy.is_file():
            paths.append(str(npy))
            n_ok += 1
        else:
            paths.append("")
        if (i + 1) % 200 == 0:
            print(f"skeleton {i + 1}/{len(table)} ok={n_ok}")
    table["skel_npy"] = paths
    table.to_csv(index_csv, index=False)
    print(f"Wrote {index_csv} skeleton={n_ok}/{len(table)}")
    return index_csv


def main() -> None:
    args = parse_args()
    index_csv = args.index or (output_dir("cache_v2") / "index.csv")
    cache_skeletons(args.data_root, index_csv, force=args.force)


if __name__ == "__main__":
    main()
