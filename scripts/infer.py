from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import COLAB_TEST, TEST_CSV, output_dir
from har.constants import NUM_CLASSES, NUM_TEST_CLIPS
from har.dataset import load_multimodal_from_test_clip, restrict_sample_modalities, to_tensors
from har.heuristic import predict_clip
from har.model import MultiModalHAR, predict_label
from har.paths import clip_id_from_path, find_test_csv, find_test_root
from har.submit import default_rows, read_test_table, write_submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=COLAB_TEST.parent)
    p.add_argument("--test-csv", type=Path, default=TEST_CSV)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_manifest(data_root: Path, test_csv: Path | None) -> pd.DataFrame:
    csv_path = test_csv or find_test_csv(data_root)
    if csv_path is not None:
        df = read_test_table(csv_path)
        if len(df) != NUM_TEST_CLIPS:
            raise SystemExit(f"{csv_path} has {len(df)} rows; expected {NUM_TEST_CLIPS}")
        return df
    test_root = find_test_root(data_root)
    if test_root is not None:
        ids = sorted(p.name for p in test_root.glob("SM_test_*") if p.is_dir())
        if len(ids) == NUM_TEST_CLIPS:
            return pd.DataFrame(
                {"path": [f"small_model_track_test/{i}/" for i in ids], "prediction": [0] * NUM_TEST_CLIPS}
            )
    return default_rows()


def resolve_clip_dir(data_root: Path, test_root: Path | None, path_key: str) -> Path | None:
    clip_id = clip_id_from_path(path_key)
    candidates = []
    if test_root is not None:
        candidates.append(test_root / clip_id)
    rel = path_key.replace("\\", "/").strip("/")
    candidates.extend(
        [
            data_root / rel,
            data_root / "small_model_track_test" / clip_id,
            data_root / "Testing" / "data" / "small_model_track_test" / clip_id,
            Path(rel),
        ]
    )
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def load_model(checkpoint: Path, device: torch.device) -> tuple[MultiModalHAR | None, dict]:
    if not checkpoint.is_file():
        return None, {"modalities": ["depth"], "crop_person": False}
    model = MultiModalHAR(num_classes=NUM_CLASSES).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()
    cfg = blob.get("train_cfg") if isinstance(blob, dict) else None
    if not cfg:
        cfg = {"modalities": ["depth"], "crop_person": False}
    return model, cfg


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    args.checkpoint = args.checkpoint or (output_dir("checkpoints") / "model.pt")
    args.out = args.out or (output_dir("submissions") / "submission.csv")
    if not args.test_csv.is_file():
        args.test_csv = None
    manifest = load_manifest(data_root, args.test_csv)
    test_root = find_test_root(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = load_model(args.checkpoint, device)
    modalities = train_cfg.get("modalities") or ["depth"]
    crop_person = bool(train_cfg.get("crop_person", False))
    print(f"checkpoint {args.checkpoint}")
    print(f"train_cfg modalities={modalities} crop_person={crop_person}")

    n_model = n_heur = n_prior = 0
    preds: list[int] = []
    for i, path_key in enumerate(manifest["path"].astype(str)):
        clip_dir = resolve_clip_dir(data_root, test_root, path_key)
        if clip_dir is None:
            preds.append(36)
            n_prior += 1
            continue
        try:
            if model is not None:
                sample = load_multimodal_from_test_clip(clip_dir, crop_person=crop_person)
                sample = restrict_sample_modalities(sample, modalities)
                preds.append(predict_label(model, to_tensors(sample), device))
                n_model += 1
            else:
                preds.append(predict_clip(clip_dir, train_layout=False))
                n_heur += 1
        except Exception as exc:
            print(f"skip {path_key}: {exc}")
            preds.append(36)
            n_prior += 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(manifest)}")

    manifest = manifest.copy()
    manifest["prediction"] = preds
    out = write_submission(manifest, args.out)
    print(
        f"Wrote {out}  ({n_model} model, {n_heur} heuristic, {n_prior} prior/no-clip)  "
        f"rows={len(manifest)}"
    )
    vc = pd.Series(preds).value_counts()
    print("prediction value_counts:")
    print(vc.head(10).to_string())
    print(f"classes {int(vc.size)} top share {float(vc.iloc[0] / max(len(preds), 1)):.3f}")


if __name__ == "__main__":
    main()
