from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.config import output_dir
from har.constants import NUM_CLASSES
from har.dataset import CachedDepthDataset, collate_clips
from har.model import MultiModalHAR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-index", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--submission", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-users", type=int, nargs="+", default=[8, 9, 23, 24])
    return p.parse_args()


@torch.no_grad()
def accuracy(model, loader, device) -> tuple[float, dict[int, int]]:
    model.eval()
    correct, n = 0, 0
    pred_counts: dict[int, int] = {}
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ("depth", "ir", "thermal", "imu", "skeleton", "radar", "mask")}
        labels = batch["label"].to(device)
        pred = model(inputs).argmax(1)
        correct += int((pred == labels).sum().item())
        n += labels.size(0)
        for p in pred.cpu().tolist():
            pred_counts[p] = pred_counts.get(p, 0) + 1
    return correct / max(n, 1), pred_counts


def recommend(val_acc: float, n_classes: int, max_frac: float) -> str:
    if n_classes <= 4 or max_frac >= 0.60:
        return "DO NOT SUBMIT — predictions collapsed."
    if val_acc < 0.10:
        return "DO NOT SUBMIT — val accuracy is near chance (0.025)."
    if val_acc < 0.20:
        return "Optional weak baseline only. Expect a similar public score, not 0.5+."
    return "OK to submit. Public LB should be in the same ballpark as val_acc."


def main() -> None:
    args = parse_args()
    cache_index = args.cache_index or (output_dir("cache") / "index.csv")
    ckpt_path = args.checkpoint or (output_dir("checkpoints") / "model.pt")
    sub_path = args.submission or (output_dir("submissions") / "submission.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    print("checkpoint", ckpt_path)
    print("  stored val_acc", blob.get("val_acc"), "epoch", blob.get("epoch"))
    print("  train_cfg", blob.get("train_cfg"))

    model = MultiModalHAR(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    model.eval()

    ds = CachedDepthDataset(cache_index, augment=False)
    users = [int(x) if x == x else -1 for x in ds.table["user_id"].tolist()]
    val_users = set(args.val_users)
    train_idx = [i for i, u in enumerate(users) if u not in val_users]
    val_idx = [i for i, u in enumerate(users) if u in val_users]
    train_acc, _ = accuracy(
        model,
        DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, collate_fn=collate_clips),
        device,
    )
    val_acc, val_pred = accuracy(
        model,
        DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, collate_fn=collate_clips),
        device,
    )
    print(f"train clips {len(train_idx)}  acc={train_acc:.3f}  ({100 * train_acc:.1f}%)")
    print(f"val   clips {len(val_idx)}  acc={val_acc:.3f}  ({100 * val_acc:.1f}%)  users={sorted(val_users)}")

    n_classes, max_frac = 0, 1.0
    if sub_path.is_file():
        df = pd.read_csv(sub_path)
        vc = df["prediction"].value_counts()
        n_classes = int(df["prediction"].nunique())
        max_frac = float(vc.iloc[0] / len(df))
        print(f"\nsubmission {sub_path}")
        print(f"  rows={len(df)}  classes_used={n_classes}/40  largest_bin={max_frac:.1%}")
        print(vc.head(10).to_string())
    else:
        print("no submission.csv yet — run infer first")

    print("VERDICT:", recommend(val_acc, n_classes, max_frac))


if __name__ == "__main__":
    main()
