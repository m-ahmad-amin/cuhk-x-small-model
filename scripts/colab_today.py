from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import COLAB_EXTRACT, COLAB_TEST, TEST_CSV, TEST_ZIP, output_dir
from har.paths import find_har_root, find_test_root, har_has_modality


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--infer-only", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=24)
    return p.parse_args()


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_test_unzipped() -> None:
    if find_test_root(COLAB_TEST.parent) is not None:
        print("test clips already at", COLAB_TEST)
        return
    if not TEST_ZIP.is_file():
        raise SystemExit(f"Missing {TEST_ZIP}")
    print("Unzipping test set to /content ...")
    with zipfile.ZipFile(TEST_ZIP) as z:
        z.extractall("/content")
    print("test root", find_test_root(Path("/content")))


def main() -> None:
    args = parse_args()
    ensure_test_unzipped()
    cache_index = output_dir("cache_v2") / "index.csv"
    ckpt = output_dir("checkpoints") / "model.pt"
    sub = output_dir("submissions") / "submission.csv"

    if args.infer_only:
        run(
            "infer.py",
            [
                "--data-root",
                "/content",
                "--test-csv",
                str(TEST_CSV),
                "--checkpoint",
                str(ckpt),
                "--out",
                str(sub),
            ],
        )
        print("DONE infer-only")
        print("submission", sub)
        return

    if not args.skip_extract:
        if find_har_root(COLAB_EXTRACT) is None:
            run("extract_depth.py", ["--out", str(COLAB_EXTRACT), "--modality", "Depth_Color"])
        else:
            print("HAR extract already present", COLAB_EXTRACT)
        har_root = find_har_root(COLAB_EXTRACT)
        if har_root is not None and not har_has_modality(har_root, "imu"):
            run("extract_depth.py", ["--out", str(COLAB_EXTRACT), "--modality", "IMU"])
        elif har_root is not None:
            print("IMU already present under", har_root)
    elif find_har_root(COLAB_EXTRACT) is not None:
        print("HAR extract already present", COLAB_EXTRACT)

    if not args.skip_cache and not cache_index.is_file():
        run(
            "cache_depth.py",
            ["--data-root", str(COLAB_EXTRACT), "--out", str(output_dir("cache_v2")), "--force"],
        )
    elif cache_index.is_file():
        print("cache index", cache_index)

    if not args.skip_train:
        run(
            "train.py",
            [
                "--cache-index",
                str(cache_index),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--out",
                str(ckpt),
            ],
        )

    run(
        "infer.py",
        [
            "--data-root",
            "/content",
            "--test-csv",
            str(TEST_CSV),
            "--checkpoint",
            str(ckpt),
            "--out",
            str(sub),
        ],
    )
    print("DONE")
    print("checkpoint", ckpt)
    print("submission", sub)


if __name__ == "__main__":
    main()
