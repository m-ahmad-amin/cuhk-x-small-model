"""Distill a multi-teacher ensemble into one student (same arch, ~45 MB)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import output_dir
from har.constants import NUM_CLASSES
from har.dataset import CachedDepthDataset, collate_clips, infer_cached_image_size, infer_cached_num_frames
from har.model import MultiModalHAR, adapt_init_state, model_from_train_cfg, model_size_mb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--teachers", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.7, help="weight on KL distill vs hard CE")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--teacher-flip", action="store_true")
    p.add_argument("--init-teacher", type=int, default=0, help="which teacher index to init student from")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(path: Path, device: torch.device) -> tuple[MultiModalHAR, dict]:
    if not path.is_file():
        raise SystemExit(f"Missing teacher {path}")
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob.get("train_cfg") if isinstance(blob, dict) else None
    if not cfg:
        raise SystemExit(f"No train_cfg in {path}")
    model = model_from_train_cfg(cfg, num_classes=NUM_CLASSES).to(device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def mixup_inputs(inputs: dict, labels: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = dict(inputs)
    for key in ("depth", "ir", "thermal", "imu", "skeleton", "radar"):
        mixed[key] = lam * mixed[key] + (1.0 - lam) * mixed[key][idx]
    mixed["mask"] = torch.maximum(mixed["mask"], mixed["mask"][idx])
    return mixed, idx, lam


def flip_depth_batch(inputs: dict) -> dict:
    out = dict(inputs)
    for key in ("depth", "ir", "thermal"):
        if torch.is_tensor(out.get(key)):
            out[key] = torch.flip(out[key], dims=[-1])
    return out


@torch.no_grad()
def teacher_ensemble_logits(
    teachers: list[MultiModalHAR], inputs: dict, teacher_flip: bool
) -> torch.Tensor:
    total = None
    for teacher in teachers:
        part = teacher(inputs)
        if teacher_flip:
            part = 0.5 * (part + teacher(flip_depth_batch(inputs)))
        total = part if total is None else total + part
    return total / max(len(teachers), 1)


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    labels_b: torch.Tensor | None = None,
    lam: float = 1.0,
) -> torch.Tensor:
    t = max(temperature, 1e-3)
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits / t, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (t * t)
    if labels_b is not None:
        ce = lam * F.cross_entropy(student_logits, labels, label_smoothing=0.05)
        ce = ce + (1.0 - lam) * F.cross_entropy(student_logits, labels_b, label_smoothing=0.05)
    else:
        ce = F.cross_entropy(student_logits, labels, label_smoothing=0.05)
    return alpha * kl + (1.0 - alpha) * ce


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = args.out or (output_dir("checkpoints") / "model_v4_distill.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    teachers: list[MultiModalHAR] = []
    cfgs: list[dict] = []
    for path in args.teachers:
        model, cfg = load_teacher(path, device)
        teachers.append(model)
        cfgs.append(cfg)
        print(f"teacher {path} width={cfg.get('width')} skel_norm={cfg.get('skel_norm')}")

    base_cfg = dict(cfgs[args.init_teacher])
    # Force student flags from first teacher (must match cache modalities).
    for key in ("skel_norm", "skel_velocity", "depth_motion", "width", "num_frames", "image_size"):
        if key in cfgs[0]:
            base_cfg[key] = cfgs[0][key]
    for i, cfg in enumerate(cfgs[1:], start=1):
        for key in ("skel_norm", "skel_velocity", "depth_motion", "width"):
            if cfg.get(key) != base_cfg.get(key):
                print(f"WARNING: teacher0 vs teacher{i} differ on {key}: {base_cfg.get(key)} vs {cfg.get(key)}")

    student = MultiModalHAR(
        num_classes=NUM_CLASSES,
        width=float(base_cfg.get("width", 1.0)),
        depth_motion=bool(base_cfg.get("depth_motion", False)),
        skel_velocity=bool(base_cfg.get("skel_velocity", False)),
        skel_norm=bool(base_cfg.get("skel_norm", False)),
    ).to(device)

    init_path = args.teachers[args.init_teacher]
    blob = torch.load(init_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    state = adapt_init_state(student, state)
    missing, unexpected = student.load_state_dict(state, strict=False)
    print(f"student init {init_path} missing={len(missing)} unexpected={len(unexpected)}")
    print(f"Model size: {model_size_mb(student):.2f} MB teachers={len(teachers)}")

    ds = CachedDepthDataset(
        args.cache_index,
        augment=True,
        use_imu=False,
        use_skeleton=True,
        use_ir=False,
    )
    num_frames = infer_cached_num_frames(args.cache_index)
    image_size = infer_cached_image_size(args.cache_index)
    train_cfg = {
        "crop_person": True,
        "modalities": ds.modalities(),
        "num_frames": int(num_frames),
        "image_size": int(image_size),
        "width": float(base_cfg.get("width", 1.0)),
        "mixup": float(args.mixup),
        "train_all": True,
        "depth_motion": bool(base_cfg.get("depth_motion", False)),
        "skel_velocity": bool(base_cfg.get("skel_velocity", False)),
        "skel_norm": bool(base_cfg.get("skel_norm", False)),
        "distill": True,
        "teachers": [str(p) for p in args.teachers],
        "temperature": float(args.temperature),
        "alpha": float(args.alpha),
    }
    print(
        f"Cache {args.cache_index} n={len(ds)} modalities={train_cfg['modalities']} "
        f"T={args.temperature} alpha={args.alpha} teacher_flip={args.teacher_flip}"
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clips,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = {
                k: batch[k].to(device)
                for k in ("depth", "ir", "thermal", "imu", "skeleton", "radar", "mask")
            }
            labels_b = None
            lam = 1.0
            if args.mixup > 0:
                inputs, idx, lam = mixup_inputs(inputs, labels, args.mixup)
                labels_b = labels[idx]
            with torch.no_grad():
                # Teachers see the same (possibly mixup) inputs; hard CE still uses labels.
                t_logits = teacher_ensemble_logits(teachers, inputs, args.teacher_flip)
            opt.zero_grad(set_to_none=True)
            s_logits = student(inputs)
            loss = distill_loss(
                s_logits,
                t_logits,
                labels,
                args.temperature,
                args.alpha,
                labels_b=labels_b,
                lam=lam,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((s_logits.argmax(1) == labels).sum().item())
            n += labels.size(0)
        sched.step()
        tr_loss = total_loss / max(n, 1)
        tr_acc = correct / max(n, 1)
        print(f"epoch {epoch:02d}  distill_loss={tr_loss:.4f} acc={tr_acc:.3f}")
        torch.save(
            {
                "model": student.state_dict(),
                "val_acc": float("nan"),
                "epoch": epoch,
                "train_cfg": train_cfg,
            },
            out,
        )
        print(f"  saved {out} epoch={epoch} train_acc={tr_acc:.3f}")
    print(f"distill done -> {out}")


if __name__ == "__main__":
    main()
