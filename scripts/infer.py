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
from har.model import (
    MultiModalHAR,
    adabn_forward,
    finish_adabn,
    model_from_train_cfg,
    predict_label,
    predict_label_ensemble,
    prepare_adabn,
)
from har.paths import clip_id_from_path, find_test_csv, find_test_root
from har.submit import default_rows, read_test_table, write_submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=COLAB_TEST.parent)
    p.add_argument("--test-csv", type=Path, default=TEST_CSV)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tta-flip", action="store_true")
    p.add_argument("--tta-time", type=int, default=1)
    p.add_argument("--tta-crop", action="store_true")
    p.add_argument("--adabn", action="store_true")
    p.add_argument("--aux-checkpoint", type=Path, default=None)
    p.add_argument("--aux2-checkpoint", type=Path, default=None)
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
        return None, {"modalities": ["depth"], "crop_person": False, "num_frames": 8, "width": 1.0}
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = blob.get("train_cfg") if isinstance(blob, dict) else None
    if not cfg:
        cfg = {"modalities": ["depth"], "crop_person": False, "num_frames": 8, "width": 1.0}
    cfg.setdefault("num_frames", 8)
    cfg.setdefault("width", 1.0)
    cfg.setdefault("image_size", 64)
    model = model_from_train_cfg(cfg, num_classes=NUM_CLASSES).to(device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()
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
    print(f"test_root {test_root}")
    if test_root is None:
        raise SystemExit(
            f"No SM_test_* under {data_root}. "
            "Extract small_model_track_test.zip to /content first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = load_model(args.checkpoint, device)
    models = [model] if model is not None else []
    if args.aux_checkpoint is not None:
        if not args.aux_checkpoint.is_file():
            raise SystemExit(f"Missing --aux-checkpoint {args.aux_checkpoint}")
        aux, aux_cfg = load_model(args.aux_checkpoint, device)
        models.append(aux)
        print(f"aux_checkpoint {args.aux_checkpoint} aux_cfg {aux_cfg}")
    if args.aux2_checkpoint is not None:
        if not args.aux2_checkpoint.is_file():
            raise SystemExit(f"Missing --aux2-checkpoint {args.aux2_checkpoint}")
        aux2, aux2_cfg = load_model(args.aux2_checkpoint, device)
        models.append(aux2)
        print(f"aux2_checkpoint {args.aux2_checkpoint} aux2_cfg {aux2_cfg}")
        print("WARNING: 3-model ensemble may exceed 100 MB Zoom budget — use for Kaggle probe only")
    modalities = train_cfg.get("modalities") or ["depth"]
    crop_person = bool(train_cfg.get("crop_person", False))
    num_frames = int(train_cfg.get("num_frames", 8))
    image_size = int(train_cfg.get("image_size", 64))
    tta_time = max(int(args.tta_time), 1)
    load_frames = num_frames * tta_time
    print(f"checkpoint {args.checkpoint}")
    print(
        f"train_cfg modalities={modalities} crop_person={crop_person} "
        f"num_frames={num_frames} image_size={image_size} width={train_cfg.get('width', 1.0)} "
        f"depth_motion={train_cfg.get('depth_motion', False)} "
        f"skel_velocity={train_cfg.get('skel_velocity', False)} "
        f"skel_norm={train_cfg.get('skel_norm', False)} "
        f"tta_flip={args.tta_flip} tta_time={tta_time} tta_crop={args.tta_crop} "
        f"adabn={args.adabn} load_frames={load_frames} n_models={len(models)}"
    )

    def load_tensors(path_key: str, n_frames: int) -> dict | None:
        clip_dir = resolve_clip_dir(data_root, test_root, path_key)
        if clip_dir is None:
            return None
        sample = load_multimodal_from_test_clip(
            clip_dir,
            crop_person=crop_person,
            num_frames=n_frames,
            image_size=image_size,
        )
        sample = restrict_sample_modalities(sample, modalities)
        return to_tensors(sample)

    if args.adabn and models:
        print("adabn pass 1/2: recalibrate BN on unlabeled test clips")
        for model in models:
            prepare_adabn(model)
        n_cal = 0
        for i, path_key in enumerate(manifest["path"].astype(str)):
            try:
                tensors = load_tensors(path_key, num_frames)
                if tensors is None:
                    continue
                for model in models:
                    adabn_forward(model, tensors, device)
                n_cal += 1
            except Exception as exc:
                print(f"adabn skip {path_key}: {exc}")
            if (i + 1) % 50 == 0:
                print(f"adabn {i + 1}/{len(manifest)}")
        for model in models:
            finish_adabn(model)
        print(f"adabn done calibrated_clips={n_cal}")

    n_model = n_heur = n_prior = 0
    preds: list[int] = []
    for i, path_key in enumerate(manifest["path"].astype(str)):
        clip_dir = resolve_clip_dir(data_root, test_root, path_key)
        if clip_dir is None:
            preds.append(36)
            n_prior += 1
            continue
        try:
            if models:
                tensors = load_tensors(path_key, load_frames)
                if tensors is None:
                    preds.append(36)
                    n_prior += 1
                    continue
                if len(models) > 1:
                    preds.append(
                        predict_label_ensemble(
                            models,
                            tensors,
                            device,
                            tta_flip=args.tta_flip,
                            win_frames=num_frames,
                            tta_crop=args.tta_crop,
                        )
                    )
                else:
                    preds.append(
                        predict_label(
                            models[0],
                            tensors,
                            device,
                            tta_flip=args.tta_flip,
                            win_frames=num_frames,
                            tta_crop=args.tta_crop,
                        )
                    )
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
