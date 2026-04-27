# Libero / Agibot Usage

## 1) Small-Subset Preprocess (for quick validation)

### Libero (128x128)
```bash
python scripts/preprocess_libero.py \
  --max-train-episodes 2 \
  --max-val-episodes 2 \
  --overwrite-latent \
  --device auto
```

### Agibot (192x256)
```bash
python scripts/preprocess_agibot.py \
  --val-json annotations_eval_small.json \
  --max-train-episodes 2 \
  --max-val-episodes 2 \
  --overwrite-latent \
  --batch-size 8 \
  --device auto
```

## 2) Full Preprocess

### Libero
```bash
python scripts/preprocess_libero.py \
  --max-train-episodes -1 \
  --max-val-episodes -1 \
  --overwrite-latent \
  --device auto
```

### Agibot
```bash
python scripts/preprocess_agibot.py \
  --val-json annotations_eval_small.json \
  --max-train-episodes -1 \
  --max-val-episodes -1 \
  --overwrite-latent \
  --batch-size 8 \
  --device auto
```

## 3) Train

### Libero
```bash
python main.py --config configs/train/libero/frame_ada.yaml
```

### Agibot
```bash
python main.py --config configs/train/agibot/frame_ada.yaml
```

## 4) Evaluate (short trajectory, comparison.mp4 only)

Set `evaluate_checkpoint` in the eval config first, then run:

### Libero
```bash
python main.py --config configs/evaluation/libero/frame_ada.yaml
```

### Agibot
```bash
python main.py --config configs/evaluation/agibot/frame_ada.yaml
```

Outputs:
- `work_dirs/irasim_infer/libero/<task>/<episode_id>_<start_frame>/comparison.mp4`
- `work_dirs/irasim_infer/agibot/<task>/<episode_id>_<start_frame>/comparison.mp4`

