# cuhk-x-small-model

CUHK-X Small Model Track: 40-class HAR from privacy sensors. CNN/GRU from scratch, ≤ 100 MB.

Working copy is **Google Drive** (`MyDrive/cuhk-x-small-model`). GitHub is a snapshot when you commit. Official zips stay in `MyDrive/Small-Model-Track/`.

## Layout

```
har/            loaders, dataset, model, submit
scripts/        extract, cache, train, infer
notebooks/      Colab entry
configs/        Drive vs Colab paths
data/           class_mapping.csv
tests/
inference.sh    verification: data_dir → CSV
checkpoints/    *.pt on Drive only (gitignored)
cache_v2/       cropped depth + IMU + skeleton npy (gitignored)
submissions/    Kaggle CSVs
```

Best public so far: **0.27860** — cropped depth + skeleton residual (`checkpoints/model_skel.pt`).

## Colab

GPU runtime. Copy **code only** to SSD (exclude `cache`, `cache_v2`, `checkpoints`, `submissions`). Outputs write back to Drive.

```bash
python tests/test_pipeline.py
python scripts/infer.py --data-root /content --checkpoint .../model_skel.pt
```

## Rules (short)

Cross-subject (train users 1–9, 16–24; test 10–11, 25–26). No RGB. No pretrained backbones. No LLMs.
