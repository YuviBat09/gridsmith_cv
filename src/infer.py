#!/usr/bin/env python
"""EPRI distribution-inspection inference library + CLI.

Library usage:
    from infer import load_model, predict, draw, conclude
    model = load_model()                      # models/best.pt, or COCO demo fallback
    dets = predict(model, image_bgr, conf=0.25, imgsz=1024, tile=False)
    vis = draw(image_bgr, dets)
    findings = conclude(dets, model.names)

CLI:
    python src/infer.py path/to/img_or_folder --out runs/infer [--conf 0.25] [--tile]
"""
from __future__ import annotations

import argparse
import csv
import colorsys
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
DEFAULT_METRICS = ROOT / "models" / "metrics.json"
CONDITION_ATTRS = ROOT / "data" / "yolo" / "condition_attributes.json"
DEMO_WEIGHTS = "yolo11n.pt"  # COCO-pretrained, auto-downloaded by ultralytics
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# RULES: edit this dict to change how detections become findings.
# All keyword matching is case-insensitive substring matching against the
# class name (and any condition attribute attached to a detection).
# ---------------------------------------------------------------------------
RULES = {
    # Class/attribute keywords that indicate a defect -> (severity, description)
    # severity: "high" => Action needed, "medium" => Review
    "issue_keywords": {
        "broken":   ("high",   "broken component"),
        "damage":   ("high",   "damaged component"),
        "crack":    ("high",   "cracked component"),
        "burn":     ("high",   "burn / flashover marks"),
        "missing":  ("high",   "missing component"),
        "contact":  ("high",   "contact with another object"),
        "leaning":  ("medium", "leaning structure"),
        "lean":     ("medium", "leaning structure"),
        "rust":     ("medium", "rust / corrosion"),
        "corro":    ("medium", "rust / corrosion"),
        "loose":    ("medium", "loose hardware"),
        "flash":    ("high",   "flashover marks"),
        "woodpecker": ("medium", "woodpecker damage"),
        "rot":      ("medium", "rot / decay"),
    },
    # Vegetation encroachment: vegetation boxes overlapping these assets
    "vegetation_keywords": ["vegetation", "tree", "veg"],
    "encroachment_targets": ["conductor", "wire", "line", "pole", "crossarm", "transformer"],
    # Overlap measure = intersection / area of the smaller box
    "encroachment_min_overlap": 0.05,
    "encroachment_severity": "high",
    # Detections with conf below this are listed for human review
    "low_conf_threshold": 0.40,
    # Number of low-confidence detections that, on their own, trigger "Review"
    "low_conf_review_count": 1,
}

STATUS_ORDER = {"OK": 0, "Review": 1, "Action needed": 2}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _read_metrics(path: Path = DEFAULT_METRICS) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _find_metric(metrics: dict, candidates) -> float | None:
    """Look up a metric under several possible key spellings (top-level or nested)."""
    flat = {}

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                flat[f"{prefix}{k}".lower()] = v
                flat.setdefault(str(k).lower(), v)

    if isinstance(metrics, dict):
        walk(metrics)
    for c in candidates:
        v = flat.get(c.lower())
        if isinstance(v, (int, float)):
            return float(v)
    return None


def load_model(path: str | Path | None = None):
    """Load trained EPRI weights if available, else a COCO demo model.

    The returned ultralytics YOLO object gets an extra attribute `info` (dict):
      demo (bool), weights (str), banner (str), map50, map50_95, metrics (dict)
    """
    from ultralytics import YOLO

    weights = Path(path) if path else DEFAULT_WEIGHTS
    demo = not weights.exists()
    if demo:
        if path:
            print(f"[infer] WARNING: weights not found at {weights}; falling back to demo model.",
                  file=sys.stderr)
        local_demo = ROOT / "models" / DEMO_WEIGHTS
        model = YOLO(str(local_demo) if local_demo.exists() else DEMO_WEIGHTS)
        info = {
            "demo": True,
            "weights": DEMO_WEIGHTS,
            "banner": "DEMO MODE: not trained on EPRI yet (using COCO-pretrained yolo11n; "
                      "classes are generic objects, not utility components).",
            "map50": None, "map50_95": None, "metrics": {},
        }
    else:
        model = YOLO(str(weights))
        metrics = _read_metrics(weights.parent / "metrics.json")
        map50 = _find_metric(metrics, ["map50", "mAP50", "metrics/mAP50(B)", "box.map50"])
        map5095 = _find_metric(metrics, ["map50_95", "map50-95", "mAP50-95", "map",
                                         "metrics/mAP50-95(B)", "box.map"])
        parts = [f"Trained EPRI model ({weights.name}, {len(model.names)} classes)"]
        if map50 is not None:
            parts.append(f"mAP50 = {map50:.3f}")
        if map5095 is not None:
            parts.append(f"mAP50-95 = {map5095:.3f}")
        info = {"demo": False, "weights": str(weights), "banner": " | ".join(parts),
                "map50": map50, "map50_95": map5095, "metrics": metrics}
    info["condition_attributes"] = _load_condition_attrs()
    model.info_dict = info
    return model


def model_info(model) -> dict:
    return getattr(model, "info_dict", {"demo": None, "banner": "unknown model"})


def _load_condition_attrs():
    try:
        return json.loads(CONDITION_ATTRS.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prediction (optionally tiled)
# ---------------------------------------------------------------------------
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, ios_thr: float = 1.01) -> list[int]:
    """Greedy NMS. Also suppresses a box whose intersection-over-smaller-area > ios_thr
    (catches duplicates nested inside a bigger box, which plain IoU misses)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        ios = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-9)
        order = order[1:][(iou <= iou_thr) & (ios <= ios_thr)]
    return keep


def _dedupe(dets: list[dict], iou_thr: float, ios_thr: float) -> list[dict]:
    """Class-wise duplicate removal (YOLO26 is NMS-free and can emit stacked duplicates)."""
    merged = []
    for cls in {d["class"] for d in dets}:
        group = [d for d in dets if d["class"] == cls]
        boxes = np.array([d["xyxy"] for d in group], dtype=float)
        scores = np.array([d["conf"] for d in group], dtype=float)
        merged += [group[i] for i in _nms(boxes, scores, iou_thr, ios_thr)]
    return merged


def _run(model, img, conf, imgsz, offset=(0, 0)):
    res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    ox, oy = offset
    for b, c, k in zip(xyxy, confs, clss):
        out.append({"class": str(res.names[int(k)]), "class_id": int(k), "conf": float(c),
                    "xyxy": [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]})
    return out


SMALL_IMAGE_PX = 2500             # below this longest side, treat input as a close-up crop
SMALL_IMAGE_SIZES = (320, 640)    # inference sizes merged for close-ups


def _tiles(h, w, tile, overlap):
    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield max(x, 0), max(y, 0)


def predict(model, image_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 1024,
            tile: bool = False, tile_size: int | None = None, overlap: float = 0.2,
            iou_merge: float = 0.5, ios_merge: float = 0.9) -> list[dict]:
    """Run detection. Returns [{class, class_id, conf, xyxy}] in original pixel coords.

    tile=True: SAHI-style slicing. The image is cut into overlapping tiles
    (tile_size, default = imgsz), each tile is predicted at full resolution, a
    full-image pass is added for large objects, and results are merged with
    class-wise NMS. Only kicks in if the image is larger than one tile.
    """
    h, w = image_bgr.shape[:2]
    if max(h, w) < SMALL_IMAGE_PX:
        # Screenshots / crops are zoomed-in vs. the 12 MP training frames: running them at
        # imgsz 1024 upscales parts beyond anything seen in training. Multi-scale + merge.
        dets = [d for s in SMALL_IMAGE_SIZES for d in _run(model, image_bgr, conf, s)]
    else:
        dets = _run(model, image_bgr, conf, imgsz)
    ts = int(tile_size or imgsz)
    if tile and max(h, w) > ts:
        for x, y in _tiles(h, w, ts, overlap):
            crop = image_bgr[y:y + ts, x:x + ts]
            dets += _run(model, crop, conf, ts, offset=(x, y))
    dets = _dedupe(dets, iou_merge, ios_merge)
    dets.sort(key=lambda d: -d["conf"])
    for d in dets:
        d["conf"] = round(d["conf"], 4)
        d["xyxy"] = [round(v, 1) for v in d["xyxy"]]
    return dets


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def class_color(name: str) -> tuple[int, int, int]:
    """Stable, distinct-ish BGR color per class name."""
    hval = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    hue = (hval % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(image_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    scale = max(h, w) / 1000.0
    thick = max(2, int(round(2 * scale)))
    font_scale = max(0.5, 0.6 * scale)
    font_thick = max(1, int(round(1.5 * scale)))
    for d in sorted(dets, key=lambda d: d["conf"]):
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = class_color(d["class"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        label = f"{d['class']} {d['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad = max(2, thick)
        ty = y1 - th - base - 2 * pad
        if ty < 0:  # no room above the box -> draw inside
            ty = y1
        tx = min(max(x1, 0), max(w - tw - 2 * pad, 0))
        cv2.rectangle(img, (tx, ty), (tx + tw + 2 * pad, ty + th + base + 2 * pad), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        txt_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(img, label, (tx + pad, ty + th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, txt_color, font_thick, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------
def _has_kw(text: str, kws) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in kws)


def _overlap_small(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def conclude(dets: list[dict], names=None, rules: dict = RULES) -> dict:
    """Turn detections into structured findings.

    Returns {status, inventory, issues, low_confidence, sentences, n_detections}.
    `names` (model.names) is accepted for context; inventory lists only classes seen.
    """
    inventory: dict[str, int] = {}
    for d in dets:
        inventory[d["class"]] = inventory.get(d["class"], 0) + 1
    inventory = dict(sorted(inventory.items(), key=lambda kv: (-kv[1], kv[0])))

    issues = []
    # 1) keyword-based defects (class name or optional attributes list on a detection)
    for d in dets:
        text = " ".join([d["class"]] + [str(a) for a in d.get("attributes", [])])
        for kw, (sev, desc) in rules["issue_keywords"].items():
            if kw.lower() in text.lower():
                issues.append({"type": "defect", "severity": sev, "class": d["class"],
                               "conf": d["conf"], "xyxy": d["xyxy"],
                               "message": f"Possible {desc}: '{d['class']}' detected "
                                          f"(confidence {d['conf']:.2f})."})
                break

    # 2) vegetation encroachment
    veg = [d for d in dets if _has_kw(d["class"], rules["vegetation_keywords"])]
    targets = [d for d in dets if _has_kw(d["class"], rules["encroachment_targets"])
               and not _has_kw(d["class"], rules["vegetation_keywords"])]
    for v in veg:
        hits = sorted({t["class"] for t in targets
                       if _overlap_small(v["xyxy"], t["xyxy"]) >= rules["encroachment_min_overlap"]})
        if hits:
            issues.append({"type": "vegetation_encroachment",
                           "severity": rules["encroachment_severity"], "class": v["class"],
                           "conf": v["conf"], "xyxy": v["xyxy"],
                           "message": f"Vegetation encroachment: vegetation overlaps "
                                      f"{', '.join(hits)} (confidence {v['conf']:.2f})."})

    low = [{"class": d["class"], "conf": d["conf"], "xyxy": d["xyxy"]}
           for d in dets if d["conf"] < rules["low_conf_threshold"]]

    if any(i["severity"] == "high" for i in issues):
        status = "Action needed"
    elif issues or len(low) >= max(1, rules["low_conf_review_count"]):
        status = "Review"
    else:
        status = "OK"

    sentences = []
    if inventory:
        inv = ", ".join(f"{n} x {c}" for c, n in inventory.items())
        sentences.append(f"Detected {len(dets)} object(s): {inv}.")
    else:
        sentences.append("No components were detected; check image quality or lower the "
                         "confidence threshold.")
    seen = set()
    for i in issues:
        if i["message"] not in seen:
            sentences.append(i["message"])
            seen.add(i["message"])
    if low:
        sentences.append(f"{len(low)} low-confidence detection(s) (< "
                         f"{rules['low_conf_threshold']:.2f}) should be checked by a human.")
    if status == "OK" and dets:
        sentences.append("No defects or encroachment flagged.")

    return {"status": status, "n_detections": len(dets), "inventory": inventory,
            "issues": issues, "low_confidence": low, "sentences": sentences}


def summarize(per_image: list[dict]) -> dict:
    """Aggregate findings across images: [{image, findings}] -> overall summary."""
    status = "OK"
    total_inv: dict[str, int] = {}
    bullets = []
    counts = {"OK": 0, "Review": 0, "Action needed": 0}
    for item in per_image:
        f = item["findings"]
        counts[f["status"]] += 1
        if STATUS_ORDER[f["status"]] > STATUS_ORDER[status]:
            status = f["status"]
        for c, n in f["inventory"].items():
            total_inv[c] = total_inv.get(c, 0) + n
        for i in f["issues"]:
            bullets.append(f"{item['image']}: {i['message']}")
        if f["low_confidence"]:
            bullets.append(f"{item['image']}: {len(f['low_confidence'])} low-confidence "
                           f"detection(s) need human review.")
    head = (f"{len(per_image)} image(s) inspected: {counts['Action needed']} need action, "
            f"{counts['Review']} need review, {counts['OK']} OK.")
    if total_inv:
        head2 = "Components found: " + ", ".join(
            f"{n} x {c}" for c, n in sorted(total_inv.items(), key=lambda kv: -kv[1])) + "."
    else:
        head2 = "No components detected."
    return {"status": status, "counts": counts, "inventory": total_inv,
            "bullets": [head, head2] + bullets}


# ---------------------------------------------------------------------------
# Batch / reporting helpers (shared with the Gradio app)
# ---------------------------------------------------------------------------
def write_reports(per_image: list[dict], out_dir: Path, info: dict | None = None):
    """Write findings.json and findings.csv (one row per image) + detections.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(per_image)
    json_path = out_dir / "findings.json"
    json_path.write_text(json.dumps({"model": {k: v for k, v in (info or {}).items()
                                               if k in ("demo", "weights", "banner",
                                                        "map50", "map50_95")},
                                     "summary": summary, "images": per_image}, indent=2))
    csv_path = out_dir / "findings.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "status", "n_detections", "inventory", "issues",
                     "n_low_confidence", "conclusions"])
        for item in per_image:
            f = item["findings"]
            wr.writerow([item["image"], f["status"], f["n_detections"],
                         "; ".join(f"{c}={n}" for c, n in f["inventory"].items()),
                         " | ".join(i["message"] for i in f["issues"]),
                         len(f["low_confidence"]), " ".join(f["sentences"])])
    det_path = out_dir / "detections.csv"
    with det_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "class", "conf", "x1", "y1", "x2", "y2"])
        for item in per_image:
            for d in item["detections"]:
                wr.writerow([item["image"], d["class"], d["conf"], *d["xyxy"]])
    return json_path, csv_path, det_path


def list_images(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run EPRI inspection inference on images.")
    ap.add_argument("source", help="image file or folder")
    ap.add_argument("--out", default="runs/infer", help="output folder")
    ap.add_argument("--weights", default=None, help="weights (default models/best.pt, else demo)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tile", action="store_true", help="tiled (SAHI-style) inference")
    ap.add_argument("--tile-size", type=int, default=None)
    args = ap.parse_args(argv)

    model = load_model(args.weights)
    info = model_info(model)
    print(f"[infer] {info['banner']}")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    imgs = list_images(Path(args.source))
    if not imgs:
        sys.exit(f"No images found at {args.source}")
    per_image = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[infer] skip unreadable {p}", file=sys.stderr)
            continue
        dets = predict(model, img, conf=args.conf, imgsz=args.imgsz, tile=args.tile,
                       tile_size=args.tile_size)
        f = conclude(dets, model.names)
        cv2.imwrite(str(out / "images" / f"{p.stem}_annotated.jpg"), draw(img, dets))
        per_image.append({"image": p.name, "detections": dets, "findings": f})
        print(f"[infer] {p.name}: {f['status']} - {' '.join(f['sentences'])}")
    j, c, _ = write_reports(per_image, out, info)
    s = summarize(per_image)
    print(f"[infer] OVERALL: {s['status']}. {s['bullets'][0]}")
    print(f"[infer] wrote {out/'images'}, {j}, {c}")


if __name__ == "__main__":
    main()
