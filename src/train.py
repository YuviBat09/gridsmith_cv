#!/usr/bin/env python
"""Train a YOLO object detector on the EPRI Distribution Inspection Imagery dataset.

Usage:
    python src/train.py --data data/yolo/dataset.yaml [--model yolo11s.pt] [--epochs 50]
                        [--imgsz 1024] [--batch auto] [--quick] [--out models]

After training, the best checkpoint is copied to <out>/best.pt and a validation
pass writes <out>/metrics.json (mAP50, mAP50-95, per-class AP50, class names,
imgsz, train date).
"""

from __future__ import annotations

import os

# Must be set before torch is imported: lets unsupported MPS ops fall back to CPU
# instead of raising NotImplementedError.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "yolo11s.pt"  # fallback; default_model() prefers yolo26s when supported
DEFAULT_OUT = PROJECT_ROOT / "models"
DEFAULT_RUNS = PROJECT_ROOT / "runs"

# Substrings of errors that indicate an MPS backend problem (unsupported op,
# placeholder storage, dtype not supported on MPS, etc.). Seeing one of these
# triggers a CPU retry.
MPS_ERROR_MARKERS = (
    "mps",
    "metal",
    "placeholder storage",
    "not implemented for",
    "not currently implemented",
    "aten::",
    "is not supported on",
)


# --------------------------------------------------------------------------- #
# Helpers shared with evaluate.py
# --------------------------------------------------------------------------- #
def pick_device(requested: str | None = None) -> str:
    """Return 'mps' if available, else 'cpu' (or honour an explicit request)."""
    import torch

    if requested and requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            print("[train] MPS requested but not available -> using CPU")
            return "cpu"
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def default_model() -> str:
    """Pick the newest small detection model the installed ultralytics supports."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

        stems = set(GITHUB_ASSETS_STEMS)
        for cand in ("yolo26s", "yolo11s", "yolov8s"):
            if cand in stems:
                return f"{cand}.pt"
    except Exception:
        pass
    return DEFAULT_MODEL


def total_ram_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) / 1024**3
    except Exception:
        try:
            import psutil

            return psutil.virtual_memory().total / 1024**3
        except Exception:
            return 16.0


def auto_batch(imgsz: int, device: str, model: str) -> int:
    """Heuristic batch size.

    Ultralytics' AutoBatch (batch=-1) only probes CUDA memory; on MPS/CPU it
    silently falls back to 16, which at imgsz=1024 can exhaust unified memory.
    So we pick a value from total RAM, image size, and model scale instead.
    """
    if device not in ("mps", "cpu"):
        return -1  # CUDA: let ultralytics AutoBatch handle it
    ram = total_ram_gb()
    base = 32 if ram >= 48 else 16 if ram >= 24 else 8  # batch at imgsz=640 for an 's' model
    stem = Path(model).stem.lower()
    scale_factor = {"n": 2.0, "s": 1.0, "m": 0.5, "l": 0.35, "x": 0.25}
    size_letter = next((c for c in reversed(stem) if c in scale_factor), "s")
    b = base * scale_factor[size_letter] * (640 / imgsz) ** 2
    if device == "cpu":
        b = min(b, 8)
    return int(max(2, min(64, round(b))))


def is_mps_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in MPS_ERROR_MARKERS)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO on EPRI distribution inspection imagery")
    p.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    p.add_argument("--model", default=None, help="Model weights or yaml (default: yolo26s.pt if supported, else yolo11s.pt)")
    p.add_argument("--epochs", type=int, default=None, help="Epochs (default 50; 5 with --quick)")
    p.add_argument("--imgsz", type=int, default=None, help="Train/val image size (default 1024; 640 with --quick)")
    p.add_argument("--batch", default="auto", help="Batch size or 'auto' (RAM/imgsz heuristic)")
    p.add_argument("--quick", action="store_true", help="Smoke test: 5 epochs, imgsz 640, 25%% of training data")
    p.add_argument("--fraction", type=float, default=None, help="Fraction of the train set to use (default 1.0; 0.25 with --quick)")
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience in epochs")
    p.add_argument("--device", default="auto", help="auto | mps | cpu | 0 ...")
    p.add_argument("--cache", default="none", choices=["none", "ram", "disk"],
                   help="Image cache (default none). Training is GPU-bound on MPS, so caching barely helps. "
                        "'disk' writes FULL-RES .npy arrays next to the images (~36MB per 12MP photo), "
                        "'ram' keeps resized copies in RAM (~imgsz^2*3 bytes/image, duplicated per worker on macOS).")
    p.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 4) // 2)),
                   help="Dataloader workers (0 = load in main thread, ultralytics' MPS default)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Where to write best.pt and metrics.json (default: models/)")
    p.add_argument("--project", default=str(DEFAULT_RUNS), help="Ultralytics runs directory (default: runs/)")
    p.add_argument("--name", default=None, help="Run name (default: epri_<timestamp>)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_train_kwargs(args: argparse.Namespace, device: str) -> dict:
    epochs = args.epochs if args.epochs is not None else (5 if args.quick else 50)
    imgsz = args.imgsz if args.imgsz is not None else (640 if args.quick else 1024)
    fraction = args.fraction if args.fraction is not None else (0.25 if args.quick else 1.0)
    if str(args.batch).lower() == "auto":
        batch = auto_batch(imgsz, device, args.model)
    else:
        batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
    cache = {"disk": "disk", "ram": True, "none": False}[args.cache]
    name = args.name or f"epri_{'quick_' if args.quick else ''}{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    return dict(
        data=str(Path(args.data).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        fraction=fraction,
        patience=args.patience,
        cache=cache,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=name,
        exist_ok=True,
        seed=args.seed,
        pretrained=True,
        # --- augmentation tuned for drone inspection photos ---
        mosaic=1.0,        # helps small objects and context variety
        close_mosaic=min(10, max(1, epochs // 5)),  # disable mosaic for final epochs
        fliplr=0.5,        # poles/hardware are left-right symmetric enough
        flipud=0.0,        # gravity matters: no upside-down poles
        degrees=0.0,       # poles are vertical; keep geometry honest
        scale=0.9,         # wide zoom range so close-up crops/screenshots look familiar
        translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # lighting/sky variation
        # --- misc ---
        amp=device not in ("mps", "cpu"),  # AMP check is CUDA-only anyway
        plots=True,
        verbose=True,
    )


def make_trainer_cls():
    """DetectionTrainer that keeps dataloader workers on MPS/CPU.

    Ultralytics forces workers=0 on MPS/CPU, which makes JPEG decoding + mosaic
    augmentation run in the training loop's thread. With high-res drone photos
    that starves the GPU, so restore the requested worker count.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class EPRITrainer(DetectionTrainer):
        def __init__(self, *a, overrides=None, **kw):
            want = (overrides or {}).get("workers")
            super().__init__(*a, overrides=overrides, **kw)
            if want:
                self.args.workers = want

    return EPRITrainer


def run_training(model_path: str, kwargs: dict):
    from ultralytics import YOLO

    model = YOLO(model_path)
    trainer = make_trainer_cls() if kwargs.get("workers", 0) > 0 else None
    model.train(trainer=trainer, **kwargs)
    return model


def train_with_fallback(model_path: str, kwargs: dict):
    """Train on the selected device; if MPS throws a known backend error, retry on CPU."""
    try:
        return run_training(model_path, kwargs), kwargs["device"]
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        if kwargs["device"] != "mps" or not is_mps_error(exc):
            raise
        traceback.print_exc()
        print(
            "\n" + "=" * 78 + "\n"
            "[train] Training failed on MPS with what looks like an MPS backend error:\n"
            f"        {type(exc).__name__}: {str(exc).splitlines()[0][:200]}\n"
            "[train] Retrying on CPU (this will be much slower). To skip MPS next time,\n"
            "        pass --device cpu.\n" + "=" * 78 + "\n",
            flush=True,
        )
        cpu_kwargs = dict(kwargs, device="cpu", batch=min(int(kwargs["batch"]) if kwargs["batch"] > 0 else 8, 8))
        return run_training(model_path, cpu_kwargs), "cpu"


def validate_and_dump(best_pt: Path, data: str, imgsz: int, device: str, out_dir: Path,
                      batch: int, extra: dict, project: str) -> dict:
    """Run val on best.pt and write metrics.json."""
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    try:
        m = model.val(data=data, imgsz=imgsz, device=device, batch=batch, plots=False, verbose=False,
                      project=project, name="_val_tmp", exist_ok=True)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps" or not is_mps_error(exc):
            raise
        print(f"[train] val failed on MPS ({exc}); retrying on CPU")
        m = model.val(data=data, imgsz=imgsz, device="cpu", batch=batch, plots=False, verbose=False,
                      project=project, name="_val_tmp", exist_ok=True)

    names = {int(k): v for k, v in model.names.items()}
    ap50 = m.box.ap50  # per evaluated class, aligned with ap_class_index
    idx = list(map(int, m.box.ap_class_index))
    per_class_ap50 = {names[i]: None for i in sorted(names)}  # None = no val instances
    per_class_ap5095 = {names[i]: None for i in sorted(names)}
    for j, ci in enumerate(idx):
        per_class_ap50[names[ci]] = round(float(ap50[j]), 4)
        per_class_ap5095[names[ci]] = round(float(m.box.ap[j]), 4)

    metrics = {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class_AP50": per_class_ap50,
        "per_class_AP50_95": per_class_ap5095,
        "class_names": [names[i] for i in sorted(names)],
        "num_classes": len(names),
        "imgsz": imgsz,
        "train_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    data = Path(args.data)
    if not data.exists():
        print(f"[train] dataset yaml not found: {data}  (run src/prepare_data.py first)", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    args.model = args.model or default_model()
    kwargs = build_train_kwargs(args, device)
    out_dir = Path(args.out).resolve()

    print("[train] configuration:")
    print(f"        model={args.model} device={device} epochs={kwargs['epochs']} imgsz={kwargs['imgsz']} "
          f"batch={kwargs['batch']} fraction={kwargs['fraction']} cache={kwargs['cache']} "
          f"patience={kwargs['patience']} quick={args.quick}")
    print(f"        PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"        out={out_dir}  runs={kwargs['project']}/{kwargs['name']}")

    t0 = time.time()
    model, used_device = train_with_fallback(args.model, kwargs)
    train_seconds = time.time() - t0

    trainer = model.trainer
    best = Path(trainer.best) if trainer and Path(trainer.best).exists() else None
    if best is None:
        last = Path(trainer.last) if trainer else None
        if last and last.exists():
            print("[train] best.pt missing, falling back to last.pt")
            best = last
        else:
            print("[train] no checkpoint produced", file=sys.stderr)
            return 1
    epochs_run = (trainer.epoch + 1) if trainer is not None else kwargs["epochs"]

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "best.pt"
    shutil.copy2(best, dst)
    print(f"[train] copied {best} -> {dst}")

    val_batch = int(kwargs["batch"]) if isinstance(kwargs["batch"], int) and kwargs["batch"] > 0 else 8
    metrics = validate_and_dump(
        dst, kwargs["data"], kwargs["imgsz"], used_device, out_dir, val_batch,
        extra={
            "model": args.model,
            "device": used_device,
            "epochs_requested": kwargs["epochs"],
            "epochs_run": epochs_run,
            "batch": kwargs["batch"],
            "fraction": kwargs["fraction"],
            "quick": args.quick,
            "train_seconds": round(train_seconds, 1),
            "sec_per_epoch": round(train_seconds / max(1, epochs_run), 1),
            "run_dir": str(trainer.save_dir) if trainer else None,
        },
        project=kwargs["project"],
    )
    shutil.rmtree(Path(kwargs["project"]) / "_val_tmp", ignore_errors=True)

    print(f"[train] wrote {out_dir / 'metrics.json'}")
    print(f"[train] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}  "
          f"({train_seconds / 60:.1f} min, {metrics['sec_per_epoch']} s/epoch on {used_device})")
    for cls, v in metrics["per_class_AP50"].items():
        print(f"        {cls:<24} AP50={'n/a' if v is None else f'{v:.4f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
