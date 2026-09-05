from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import COLAB_EXTRACT, TRAIN_ZIP_DIR
from har.paths import find_har_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", type=Path, default=TRAIN_ZIP_DIR)
    p.add_argument("--out", type=Path, default=COLAB_EXTRACT)
    p.add_argument("--modality", default="Depth_Color")
    return p.parse_args()


def ensure_7z() -> str:
    exe = shutil.which("7z") or shutil.which("7za")
    if exe:
        return exe
    subprocess.check_call(["apt-get", "-qq", "update"])
    subprocess.check_call(["apt-get", "-qq", "install", "-y", "p7zip-full"])
    exe = shutil.which("7z") or shutil.which("7za")
    if not exe:
        raise SystemExit("7z not found. Install p7zip-full.")
    return exe


def find_har_zip(zip_dir: Path) -> Path:
    zip_dir = Path(zip_dir)
    candidate = zip_dir / "HAR.zip"
    if candidate.is_file():
        return candidate
    hits = list(zip_dir.glob("**/HAR.zip"))
    if hits:
        return hits[0]
    raise SystemExit(f"HAR.zip not found under {zip_dir}")


def list_sample_paths(exe: str, zip_path: Path, needle: str, limit: int = 15) -> list[str]:
    proc = subprocess.run(
        [exe, "l", "-r", f"-i!*{needle}*", str(zip_path)],
        cwd=str(zip_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    hits = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if needle.lower() in line.lower() and ("." in line or "/" in line or "\\" in line):
            hits.append(line)
            if len(hits) >= limit:
                break
    if not hits:
        print(proc.stdout[-2000:] if proc.stdout else proc.stderr[-2000:])
    return hits


def _extract_has_files(out: Path, modality: str) -> bool:
    key = modality.lower()
    if key in {"imu", "skeleton", "radar"}:
        n = sum(1 for _ in out.rglob("*") if _.suffix.lower() in {".csv", ".json", ".npy"})
        print(f"  table/json files so far: {n}")
        return n > 0
    pngs = list(out.rglob("*.png"))
    print(f"  pngs so far: {len(pngs)}")
    return bool(pngs)


def extract(zip_dir: Path, out: Path, modality: str) -> Path:
    zip_path = find_har_zip(zip_dir)
    out.mkdir(parents=True, exist_ok=True)
    exe = ensure_7z()
    print(f"Listing archive for {modality!r} ...")
    samples = list_sample_paths(exe, zip_path, modality)
    if not samples and modality.lower() in {"depth", "depth_color"}:
        samples = list_sample_paths(exe, zip_path, "Depth")
    print("sample paths:")
    for s in samples[:10]:
        print(" ", s)
    print(f"Extracting {modality} from {zip_path} -> {out}")
    needles = [modality]
    if modality.lower() in {"depth", "depth_color"}:
        needles.extend(["Depth_Color", "Depth"])
    for needle in needles:
        inc = f"-i!*{needle}*"
        cmd = [exe, "x", str(zip_path), f"-o{out}", "-y", "-r", inc]
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(zip_path.parent))
        if _extract_has_files(out, modality):
            break
    har_root = find_har_root(out)
    if har_root is None:
        direct = out / modality
        if direct.is_dir():
            nested = out / "HAR" / "data" / modality
            nested.parent.mkdir(parents=True, exist_ok=True)
            if not nested.exists():
                shutil.move(str(direct), str(nested))
            har_root = find_har_root(out)
    if har_root is None:
        raise SystemExit(
            f"Extract finished but HAR tree not found under {out}. "
            "Paste the sample paths printed above."
        )
    n_png = sum(1 for _ in har_root.rglob("*.png"))
    n_csv = sum(1 for _ in har_root.rglob("*.csv"))
    print(f"OK har_root={har_root} pngs={n_png} csvs={n_csv}")
    return har_root


def main() -> None:
    args = parse_args()
    extract(args.zip_dir, args.out, args.modality)


if __name__ == "__main__":
    main()
