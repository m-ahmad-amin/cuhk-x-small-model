from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import output_dir
from har.constants import NUM_CLASSES, NUM_FRAMES
from har.dataset import (
    CachedDepthDataset,
    HARClipDataset,
    collate_clips,
    index_train_clips,
    infer_cached_image_size,
    infer_cached_num_frames,
)
from har.model import MultiModalHAR, adapt_init_state, model_size_mb
from har.paths import find_har_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--cache-index", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--val-users", type=int, nargs="+", default=[8, 9, 23, 24])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth-only", action="store_true")
    p.add_argument("--use-skeleton", action="store_true")
    p.add_argument("--use-imu", action="store_true")
    p.add_argument("--use-ir", action="store_true")
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--freeze-depth", action="store_true")
    p.add_argument("--freeze-skeleton", action="store_true")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--train-all", action="store_true")
    p.add_argument("--depth-motion", action="store_true")
    p.add_argument("--skel-velocity", action="store_true")
    p.add_argument("--skel-norm", action="store_true")
    p.add_argument("--balanced", action="store_true")
    return p.parse_args()


def cache_modality_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    use_skel = bool(args.use_skeleton)
    use_imu = bool(args.use_imu)
    use_ir = bool(args.use_ir)
    if args.depth_only:
        use_imu = False
        use_ir = False
        if not args.use_skeleton:
            use_skel = False
    if not args.depth_only and not args.use_skeleton and not args.use_imu and not args.use_ir:
        use_imu = True
    return use_imu, use_skel, use_ir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_and_users(ds: Dataset) -> tuple[list[int], list[int]]:
    if isinstance(ds, CachedDepthDataset):
        labels = [int(x) for x in ds.table["label"].tolist()]
        users = [int(x) if x == x else -1 for x in ds.table["user_id"].tolist()]
        return labels, users
    if isinstance(ds, HARClipDataset):
        labels = [int(c.label) if c.label is not None else -1 for c in ds.clips]
        users = [int(c.user_id) if c.user_id is not None else -1 for c in ds.clips]
        return labels, users
    raise TypeError(type(ds))


def split_by_user(users: list[int], val_users: set[int]) -> tuple[list[int], list[int]]:
    train_idx, val_idx = [], []
    for i, uid in enumerate(users):
        (val_idx if uid in val_users else train_idx).append(i)
    return train_idx, val_idx


def mixup_inputs(
    inputs: dict, labels: torch.Tensor, alpha: float
) -> tuple[dict, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=labels.device)
    mixed = dict(inputs)
    for key in ("depth", "ir", "thermal", "imu", "skeleton", "radar"):
        mixed[key] = lam * mixed[key] + (1.0 - lam) * mixed[key][idx]
    mixed["mask"] = torch.maximum(mixed["mask"], mixed["mask"][idx])
    return mixed, idx, lam


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool,
    freeze_depth: bool = False,
    freeze_skeleton: bool = False,
    mixup: float = 0.0,
) -> tuple[float, float]:
    model.train(train)
    if freeze_depth:
        model.depth_enc.eval()
        model.depth_head.eval()
    if freeze_skeleton:
        model.skel_enc.eval()
        model.skel_delta.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        labels = batch["label"].to(device)
        inputs = {k: batch[k].to(device) for k in ("depth", "ir", "thermal", "imu", "skeleton", "radar", "mask")}
        idx = None
        lam = 1.0
        if train and mixup > 0:
            inputs, idx, lam = mixup_inputs(inputs, labels, mixup)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if idx is not None:
            loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[idx])
        else:
            loss = criterion(logits, labels)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        n += labels.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = args.out or (output_dir("checkpoints") / "model.pt")

    if args.cache_index is not None:
        use_imu, use_skel, use_ir = cache_modality_flags(args)
        ds = CachedDepthDataset(
            args.cache_index,
            augment=True,
            use_imu=use_imu,
            use_skeleton=use_skel,
            use_ir=use_ir,
        )
        num_frames = args.num_frames or infer_cached_num_frames(args.cache_index)
        image_size = infer_cached_image_size(args.cache_index)
        train_cfg = {
            "crop_person": True,
            "modalities": ds.modalities(),
            "num_frames": int(num_frames),
            "image_size": int(image_size),
            "width": float(args.width),
            "mixup": float(args.mixup),
            "train_all": bool(args.train_all),
            "depth_motion": bool(args.depth_motion),
            "skel_velocity": bool(args.skel_velocity),
            "skel_norm": bool(args.skel_norm),
        }
        print(
            f"Cache dataset {args.cache_index} n={len(ds)} "
            f"modalities={train_cfg['modalities']} num_frames={num_frames} "
            f"image_size={image_size} width={args.width} mixup={args.mixup} "
            f"train_all={args.train_all} depth_motion={args.depth_motion} "
            f"skel_velocity={args.skel_velocity} skel_norm={args.skel_norm}"
        )
    else:
        if args.data_root is None:
            raise SystemExit("Pass --cache-index or --data-root")
        har_root = find_har_root(args.data_root)
        if har_root is None:
            raise SystemExit(f"No HAR tree under {args.data_root}")
        clips = index_train_clips(har_root)
        num_frames = args.num_frames or NUM_FRAMES
        image_size = 112
        ds = HARClipDataset(
            clips,
            train_layout=True,
            augment=True,
            crop_person=True,
            num_frames=num_frames,
            image_size=image_size,
        )
        train_cfg = {
            "crop_person": True,
            "modalities": ["depth", "imu", "skeleton"],
            "num_frames": int(num_frames),
            "image_size": int(image_size),
            "width": float(args.width),
            "mixup": float(args.mixup),
            "depth_motion": bool(args.depth_motion),
            "skel_velocity": bool(args.skel_velocity),
            "skel_norm": bool(args.skel_norm),
        }
        print(
            f"Live dataset {har_root} n={len(ds)} num_frames={num_frames} "
            f"image_size={image_size} width={args.width} mixup={args.mixup}"
        )

    labels, users = labels_and_users(ds)
    train_idx, val_idx = split_by_user(users, set(args.val_users))
    if args.train_all:
        train_idx = list(range(len(ds)))
        val_idx = []
        print(f"train-all n={len(train_idx)} (save last epoch, no val selection)")
    else:
        if not train_idx:
            train_idx = list(range(len(ds)))
        if not val_idx:
            val_idx = train_idx[: max(1, len(train_idx) // 10)]
        print(f"split train={len(train_idx)} val={len(val_idx)} users_val={args.val_users}")

    if args.cache_index is not None:
        use_imu, use_skel, use_ir = cache_modality_flags(args)
        train_view = CachedDepthDataset(
            args.cache_index,
            augment=True,
            use_imu=use_imu,
            use_skeleton=use_skel,
            use_ir=use_ir,
        )
        val_view = CachedDepthDataset(
            args.cache_index,
            augment=False,
            use_imu=use_imu,
            use_skeleton=use_skel,
            use_ir=use_ir,
        )
    else:
        train_view = HARClipDataset(
            ds.clips,
            train_layout=True,
            augment=True,
            crop_person=True,
            num_frames=num_frames,
            image_size=image_size,
        )
        val_view = HARClipDataset(
            ds.clips,
            train_layout=True,
            augment=False,
            crop_person=True,
            num_frames=num_frames,
            image_size=image_size,
        )

    train_loader = DataLoader(
        Subset(train_view, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clips,
        pin_memory=True,
    )
    val_loader = None
    if val_idx:
        val_loader = DataLoader(
            Subset(val_view, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_clips,
            pin_memory=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = MultiModalHAR(
        num_classes=NUM_CLASSES,
        width=args.width,
        depth_motion=args.depth_motion,
        skel_velocity=args.skel_velocity,
        skel_norm=args.skel_norm,
    ).to(device)
    if args.init is not None:
        if not args.init.is_file():
            raise SystemExit(f"Missing --init checkpoint {args.init}")
        blob = torch.load(args.init, map_location=device, weights_only=False)
        init_cfg = blob.get("train_cfg") if isinstance(blob, dict) else None
        if init_cfg is not None:
            init_width = float(init_cfg.get("width", 1.0))
            if abs(init_width - float(args.width)) > 1e-6:
                raise SystemExit(f"--width {args.width} != init width {init_width}")
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        state = adapt_init_state(model, state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init {args.init} missing={len(missing)} unexpected={len(unexpected)}")
    if args.skel_velocity and not args.freeze_depth:
        print("WARNING: --skel-velocity without --freeze-depth can drift the 0.353 depth encoder")
    if args.freeze_depth and args.depth_motion:
        print("WARNING: --freeze-depth with --depth-motion leaves motion channels at 0")
    if args.freeze_depth:
        for name, param in model.named_parameters():
            if name.startswith("depth_enc") or name.startswith("depth_head"):
                param.requires_grad = False
        print("froze depth_enc + depth_head")
    if args.freeze_skeleton:
        for name, param in model.named_parameters():
            if name.startswith("skel_enc") or name.startswith("skel_delta"):
                param.requires_grad = False
        print("froze skel_enc + skel_delta")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Model size: {model_size_mb(model):.2f} MB (limit 100) trainable_tensors={len(trainable)}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    if args.balanced:
        counts = np.bincount([labels[i] for i in train_idx], minlength=NUM_CLASSES).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (NUM_CLASSES * counts)
        weights = weights / weights.mean()
        crit = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=0.05,
        )
        print(f"balanced CE weights min={weights.min():.3f} max={weights.max():.3f}")
    else:
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_acc = -1.0
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            opt,
            crit,
            device,
            True,
            freeze_depth=args.freeze_depth,
            freeze_skeleton=args.freeze_skeleton,
            mixup=args.mixup,
        )
        if val_loader is not None:
            va_loss, va_acc = run_epoch(
                model,
                val_loader,
                opt,
                crit,
                device,
                False,
                freeze_depth=args.freeze_depth,
                freeze_skeleton=args.freeze_skeleton,
            )
        else:
            va_loss, va_acc = float("nan"), float("nan")
        sched.step()
        if val_loader is not None:
            print(
                f"epoch {epoch:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
            )
        else:
            print(f"epoch {epoch:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}")
        should_save = args.train_all or va_acc >= best_acc
        if should_save:
            if val_loader is not None:
                best_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": va_acc,
                    "epoch": epoch,
                    "train_cfg": train_cfg,
                },
                out,
            )
            if args.train_all:
                print(f"  saved {out} epoch={epoch} train_acc={tr_acc:.3f}")
            else:
                print(f"  saved {out} val_acc={va_acc:.3f}")
    if args.train_all:
        print(f"last epoch -> {out}")
    else:
        print(f"best val_acc={best_acc:.3f} -> {out}")


if __name__ == "__main__":
    main()
