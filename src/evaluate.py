#!/usr/bin/env python
"""Evaluate a trained YOLO detector: per-class table + confusion matrix.

Usage:
    python src/evaluate.py --data data/yolo/dataset.yaml [--weights models/best.pt]
                           [--imgsz 1024] [--split val] [--conf 0.001] [--out models]

Writes <out>/confusion_matrix.png, <out>/confusion_matrix_normalized.png and
<out>/eval_<split>.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DEFAULT_OUT, is_mps_error, pick_device  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate YOLO weights on a YOLO dataset")
    p.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    p.add_argument("--weights", default=str(DEFAULT_OUT / "best.pt"))
    p.add_argument("--imgsz", type=int, default=None, help="Default: imgsz from metrics.json / training args, else 1024")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for mAP (keep low)")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Directory for confusion matrix + eval json")
    return p.parse_args(argv)


def resolve_imgsz(args, model) -> int:
    if args.imgsz:
        return args.imgsz
    mfile = Path(args.weights).resolve().parent / "metrics.json"
    if mfile.exists():
        try:
            return int(json.loads(mfile.read_text())["imgsz"])
        except Exception:
            pass
    try:
        return int(model.ckpt["train_args"]["imgsz"])
    except Exception:
        return 1024


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[eval] weights not found: {weights}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgsz = resolve_imgsz(args, model)
    device = pick_device(args.device)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="epri_eval_"))

    val_kw = dict(data=str(Path(args.data).resolve()), imgsz=imgsz, split=args.split, batch=args.batch,
                  conf=args.conf, iou=args.iou, plots=True, verbose=False,
                  project=str(tmp), name="val", exist_ok=True)
    print(f"[eval] weights={weights} split={args.split} imgsz={imgsz} device={device}")
    try:
        m = model.val(device=device, **val_kw)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps" or not is_mps_error(exc):
            raise
        print(f"[eval] MPS error ({str(exc).splitlines()[0][:160]}); retrying on CPU")
        device = "cpu"
        m = model.val(device=device, **val_kw)

    names = {int(k): v for k, v in model.names.items()}
    nt = getattr(m, "nt_per_class", None)
    ni = getattr(m, "nt_per_image", None)
    evaluated = {int(c): j for j, c in enumerate(m.box.ap_class_index)}

    rows = []
    hdr = f"{'class':<26}{'images':>8}{'inst':>8}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for ci in sorted(names):
        n_inst = int(nt[ci]) if nt is not None and ci < len(nt) else None
        n_img = int(ni[ci]) if ni is not None and ci < len(ni) else None
        if ci in evaluated:
            p, r, ap50, ap = (float(x) for x in m.box.class_result(evaluated[ci]))
        else:
            p = r = ap50 = ap = None
        rows.append(dict(cls=names[ci], images=n_img, instances=n_inst, precision=p, recall=r, AP50=ap50, AP50_95=ap))
        f = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"  # noqa: E731
        print(f"{names[ci][:25]:<26}{n_img if n_img is not None else '-':>8}{n_inst if n_inst is not None else '-':>8}"
              f"{f(p)}{f(r)}{f(ap50)}{f(ap)[:-1] if ap is None else f'{ap:9.3f}'}")
    print("-" * len(hdr))
    print(f"{'all':<26}{'':>8}{int(sum(nt)) if nt is not None else '':>8}"
          f"{m.box.mp:8.3f}{m.box.mr:8.3f}{m.box.map50:8.3f}{m.box.map:9.3f}\n")

    saved = []
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(m.save_dir) / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            saved.append(str(out_dir / fname))
    report = {
        "weights": str(weights.resolve()),
        "data": val_kw["data"],
        "split": args.split,
        "imgsz": imgsz,
        "device": device,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": rows,
        "confusion_matrix": m.confusion_matrix.matrix.astype(int).tolist(),  # rows=pred, cols=true (+background)
        "confusion_matrix_labels": [names[i] for i in sorted(names)] + ["background"],
    }
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    for s in saved:
        print(f"[eval] saved {s}")
    print(f"[eval] saved {out_dir / f'eval_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
