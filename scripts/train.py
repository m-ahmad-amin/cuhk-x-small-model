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
from har.constants import NUM_CLASSES
from har.dataset import CachedDepthDataset, HARClipDataset, collate_clips, index_train_clips
from har.model import MultiModalHAR, model_size_mb
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
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--freeze-depth", action="store_true")
    p.add_argument("--freeze-skeleton", action="store_true")
    return p.parse_args()


def cache_modality_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    use_skel = bool(args.use_skeleton)
    use_imu = bool(args.use_imu)
    if args.depth_only:
        use_imu = False
        if not args.use_skeleton:
            use_skel = False
    if not args.depth_only and not args.use_skeleton and not args.use_imu:
        use_imu = True
    return use_imu, use_skel


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


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool,
    freeze_depth: bool = False,
    freeze_skeleton: bool = False,
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
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
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
        use_imu, use_skel = cache_modality_flags(args)
        ds = CachedDepthDataset(
            args.cache_index, augment=True, use_imu=use_imu, use_skeleton=use_skel
        )
        train_cfg = {"crop_person": True, "modalities": ds.modalities()}
        print(f"Cache dataset {args.cache_index} n={len(ds)} modalities={train_cfg['modalities']}")
    else:
        if args.data_root is None:
            raise SystemExit("Pass --cache-index or --data-root")
        har_root = find_har_root(args.data_root)
        if har_root is None:
            raise SystemExit(f"No HAR tree under {args.data_root}")
        clips = index_train_clips(har_root)
        ds = HARClipDataset(clips, train_layout=True, augment=True, crop_person=True)
        train_cfg = {"crop_person": True, "modalities": ["depth", "imu", "skeleton"]}
        print(f"Live dataset {har_root} n={len(ds)}")

    labels, users = labels_and_users(ds)
    train_idx, val_idx = split_by_user(users, set(args.val_users))
    if not train_idx:
        train_idx = list(range(len(ds)))
    if not val_idx:
        val_idx = train_idx[: max(1, len(train_idx) // 10)]
    print(f"split train={len(train_idx)} val={len(val_idx)} users_val={args.val_users}")

    if args.cache_index is not None:
        use_imu, use_skel = cache_modality_flags(args)
        train_view = CachedDepthDataset(
            args.cache_index, augment=True, use_imu=use_imu, use_skeleton=use_skel
        )
        val_view = CachedDepthDataset(
            args.cache_index, augment=False, use_imu=use_imu, use_skeleton=use_skel
        )
    else:
        train_view = HARClipDataset(ds.clips, train_layout=True, augment=True, crop_person=True)
        val_view = HARClipDataset(ds.clips, train_layout=True, augment=False, crop_person=True)

    train_loader = DataLoader(
        Subset(train_view, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clips,
        pin_memory=True,
    )
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
    model = MultiModalHAR(num_classes=NUM_CLASSES).to(device)
    if args.init is not None:
        if not args.init.is_file():
            raise SystemExit(f"Missing --init checkpoint {args.init}")
        blob = torch.load(args.init, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init {args.init} missing={len(missing)} unexpected={len(unexpected)}")
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
        )
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
        sched.step()
        print(
            f"epoch {epoch:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if va_acc >= best_acc:
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
            print(f"  saved {out} val_acc={va_acc:.3f}")
    print(f"best val_acc={best_acc:.3f} -> {out}")


if __name__ == "__main__":
    main()
