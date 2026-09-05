from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import COLAB_EXTRACT, output_dir
from har.dataset import index_train_clips
from har.loaders import load_frames, load_imu
from har.paths import find_har_root, sibling_modality_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=COLAB_EXTRACT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-crop-person", action="store_true")
    return p.parse_args()


def cache_clips(data_root: Path, out: Path, force: bool = False, crop_person: bool = True) -> Path:
    har_root = find_har_root(data_root)
    if har_root is None:
        raise SystemExit(f"No HAR tree under {data_root}")
    clips = index_train_clips(har_root)
    if not clips:
        raise SystemExit("Indexed 0 clips")
    depth_dir = out / "depth"
    imu_dir = out / "imu"
    depth_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    rows = []
    n_imu = 0
    for i, ref in enumerate(clips):
        npy = depth_dir / f"{i:06d}.npy"
        imu_npy = imu_dir / f"{i:06d}.npy"
        if force or not npy.is_file():
            depth, ok = load_frames(ref.clip_dir, crop_person=crop_person)
            if not ok:
                continue
            np.save(npy, depth.astype(np.float16))
        if not npy.is_file():
            continue
        imu_path = ""
        imu_folder = sibling_modality_dir(ref.clip_dir, "imu")
        if imu_folder is not None:
            if force or not imu_npy.is_file():
                imu, imu_ok = load_imu(imu_folder)
                if imu_ok:
                    np.save(imu_npy, imu.astype(np.float16))
            if imu_npy.is_file():
                imu_path = str(imu_npy)
                n_imu += 1
        rows.append(
            {
                "npy": str(npy),
                "imu_npy": imu_path,
                "label": ref.label,
                "user_id": ref.user_id,
                "path_key": ref.path_key,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"cached {i + 1}/{len(clips)} imu={n_imu}")
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npy", "imu_npy", "label", "user_id", "path_key"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {index_path} n={len(rows)} imu={n_imu} crop_person={crop_person}")
    return index_path


def main() -> None:
    args = parse_args()
    out = args.out or (output_dir("cache_v2"))
    cache_clips(args.data_root, out, force=args.force, crop_person=not args.no_crop_person)


if __name__ == "__main__":
    main()
