#!/usr/bin/env python
"""Prepare the EPRI Distribution Inspection Imagery dataset (local circuits only) for Ultralytics YOLO.

Labels: Kaggle dexterlewis/epri-distribution-inspection-imagery -> Overhead-Distribution-Labels.csv
  columns "Label","External ID"; Label is a *Python-literal* dict (single quotes), e.g.
  {'objects': [{'value': 'insulator', 'polygon': [{'x':..,'y':..},..], 'classifications':
  [{'value': 'obscured/truncated', 'answer': {'value': 'obscured'}}]}], 'classifications': [...], 'relationships': []}
  Geometry is 'polygon' or 'line' (conductors/wires). Also accepts Labelbox JSON/NDJSON and COCO JSON.
  External IDs look like "5 (123).JPG" -> the leading number is the circuit.

Usage:
  python src/prepare_data.py --raw data/raw --labels data/raw/Overhead-Distribution-Labels.csv --out data/yolo
"""
import argparse, ast, csv, json, random, re, shutil, sys, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


# ----------------------------------------------------------------------------- images
def downscale(p: Path, dst: Path, max_side: int) -> Path:
    """Cached same-aspect downscale of p to dst (longest side <= max_side); returns the path to use."""
    if dst.exists() and dst.stat().st_mtime >= p.stat().st_mtime:
        return dst
    with Image.open(p) as im:
        if max(im.size) <= max_side:
            return p
        im.draft("RGB", (max_side, max_side))  # fast JPEG DCT-domain pre-shrink
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=92)
    return dst


def index_images(raw: Path, out: Path):
    """Extract Circuit*.zip into raw/images/<circuit>/ (flattened), then index every image under raw."""
    for zp in sorted(raw.glob("*.zip")):
        dest = raw / "images" / norm(zp.stem)
        if (dest / ".extracted").exists():
            continue
        print(f"extracting {zp.name} -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            for m in zf.infolist():
                name = Path(m.filename).name  # flatten nested folders (also blocks zip-slip)
                if m.is_dir() or name.startswith("._") or Path(name).suffix.lower() not in IMG_EXT:
                    continue
                with zf.open(m) as src, open(dest / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        (dest / ".extracted").touch()
    index, out = {}, out.resolve()
    for p in sorted(raw.rglob("*")):
        if p.suffix.lower() in IMG_EXT and out not in p.absolute().parents and not p.name.startswith("._"):
            if p.name.lower() in index and index[p.name.lower()] != p:
                print(f"  warn: duplicate image name {p.name}, keeping {index[p.name.lower()]}")
                continue
            index[p.name.lower()] = p
    return index


# ----------------------------------------------------------------------------- labels
def _loads(s):
    if isinstance(s, (dict, list)) or s is None:
        return s
    s = s.strip()
    if not s or s.lower() in ("skip", "{}", "none"):
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)  # the EPRI CSV stores Python reprs (single quotes)


def _attrs(classifications):
    """Labelbox classifications (v1 or NDJSON) -> {attr: value}."""
    out = {}
    for c in classifications or []:
        key = c.get("value") or c.get("title") or c.get("name")
        ans = c.get("answer", c.get("answers", c.get("radio_answer", c.get("checklist_answers", c.get("text_answer")))))
        vals = ans if isinstance(ans, list) else [ans]
        vals = [(a.get("value") or a.get("title") or a.get("name") or a.get("content")) if isinstance(a, dict) else a
                for a in vals if a is not None]
        if key and vals:
            out[norm(key)] = "+".join(sorted(norm(v) for v in vals))
    return out


def _pts(o):
    b = o.get("bbox") or o.get("bounding_box")
    if b:
        x, y, w, h = b["left"], b["top"], b["width"], b["height"]
        return [(x, y), (x + w, y + h)], "bbox"
    for kind in ("polygon", "line"):
        g = o.get(kind)
        if g:
            return [(p["x"], p["y"]) for p in g], kind
    return None, None


def _obj(o):
    pts, kind = _pts(o)
    name = o.get("value") or o.get("title") or o.get("name")
    return dict(cls=norm(name), pts=pts, kind=kind, attrs=_attrs(o.get("classifications"))) if pts and name else None


def _base(ext_id):  # "5 (1).JPG", "5%20(1).JPG", "https://.../5%20(1).JPG?sig=.." -> "5 (1).JPG"
    return Path(unquote(str(ext_id).split("?")[0])).name


def _rec(ext_id, label, size=None):
    objs = [x for x in map(_obj, label.get("objects", [])) if x]
    return dict(name=_base(ext_id), objects=objs, img_attrs=_attrs(label.get("classifications")), size=size)


def load_labels(path: Path):
    """Auto-detect: Labelbox v1 CSV/JSON, Labelbox NDJSON (v2 export), or COCO JSON."""
    if path.suffix.lower() == ".csv" or path.read_bytes()[:64].lstrip(b"\xef\xbb\xbf \r\n\t")[:1] not in (b"{", b"["):
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return [_rec(r.get("External ID") or r.get("external_id"), _loads(r.get("Label")) or {}) for r in rows], "labelbox-v1-csv"
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # NDJSON: one data row per line
        data = [json.loads(l) for l in text.splitlines() if l.strip()]
    if isinstance(data, dict) and "annotations" in data and "images" in data:  # COCO
        cats = {c["id"]: c["name"] for c in data["categories"]}
        by_img = defaultdict(list)
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]].append(dict(cls=norm(cats[a["category_id"]]), pts=[(x, y), (x + w, y + h)], kind="bbox",
                                              attrs={norm(k): norm(v) for k, v in (a.get("attributes") or {}).items()}))
        return [dict(name=_base(i["file_name"]), objects=by_img[i["id"]], img_attrs={}, size=(i.get("width"), i.get("height")))
                for i in data["images"]], "coco"
    data = data if isinstance(data, list) else [data]
    recs, fmt = [], "labelbox-v1-json"
    for r in data:
        if "data_row" in r:  # NDJSON v2: {data_row:{external_id}, media_attributes, projects:{id:{labels:[{annotations}]}}}
            fmt, ma = "labelbox-ndjson", r.get("media_attributes") or {}
            ann = [l["annotations"] for p in (r.get("projects") or {}).values() for l in p.get("labels", [])]
            label = {"objects": [o for a in ann for o in a.get("objects", [])],
                     "classifications": [c for a in ann for c in a.get("classifications", [])]}
            recs.append(_rec(r["data_row"].get("external_id") or r["data_row"].get("row_data", ""), label,
                             (ma.get("width"), ma.get("height")) if ma.get("width") else None))
        else:
            recs.append(_rec(r.get("External ID") or r.get("external_id"), _loads(r.get("Label")) or {}))
    return recs, fmt


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=20, help="drop classes with fewer instances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-condition", action="store_true", help='class name += "__<attr values>", e.g. insulator__obscured')
    ap.add_argument("--drop-classes", default="", help="comma-separated classes to drop, e.g. conductor,other_wire")
    ap.add_argument("--min-box-frac", type=float, default=0.01,
                    help="pad thin boxes (e.g. from 'line' wires) to at least this fraction of max(img w,h)")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="write downscaled copies (longest side, px) to raw/resized_<N>/ and train on those; "
                         "speeds up data loading. 0 = symlink the full-res originals")
    a = ap.parse_args()

    index = index_images(a.raw, a.out)
    recs, fmt = load_labels(a.labels)
    print(f"label format: {fmt}; {len(recs)} labeled images in file; {len(index)} images found locally")

    matched, seen = [], set()
    for r in recs:
        p = index.get(r["name"].lower())
        if p and r["objects"] and r["name"].lower() not in seen:
            seen.add(r["name"].lower())
            matched.append((p, r))
    print(f"matched: {len(matched)} | labels w/o local image: {len(recs) - len(matched)} | "
          f"local images w/o labels: {len(set(index) - seen)}")
    if not matched:
        sys.exit("no labeled images matched local files -- check --raw / --labels and filenames")

    drop = {norm(c) for c in a.drop_classes.split(",") if c.strip()}
    cond = defaultdict(lambda: defaultdict(Counter))
    img_cond = defaultdict(Counter)
    for _, r in matched:
        for k, v in r["img_attrs"].items():
            img_cond[k][v] += 1
        for o in r["objects"]:
            for k, v in o["attrs"].items():
                cond[o["cls"]][k][v] += 1
            o["name"] = o["cls"] + ("__" + "_".join(o["attrs"][k] for k in sorted(o["attrs"])) if a.merge_condition and o["attrs"] else "")
    counts = Counter(o["name"] for _, r in matched for o in r["objects"])
    print("objects per class:\n" + "\n".join(f"  {n:<40}{c}" for n, c in counts.most_common()))
    names = [n for n, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= a.min_count and n.split("__")[0] not in drop]
    dropped = sorted(set(counts) - set(names))
    if dropped:
        print(f"dropped classes (<{a.min_count} or --drop-classes): {dropped}")
    cid = {n: i for i, n in enumerate(names)}

    # split per circuit (folder or "<circuit> (n).JPG" prefix) so both circuits appear in train and val
    rng, groups = random.Random(a.seed), defaultdict(list)
    for p, r in sorted(matched, key=lambda x: x[1]["name"]):
        m = re.match(r"(\d+)\s*\(", p.name)
        groups[m.group(1) if m else p.parent.name].append((p, r))
    split = {}
    for g, items in sorted(groups.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * a.val_frac)) if len(items) > 1 and a.val_frac > 0 else 0
        for i, (p, r) in enumerate(items):
            split[p] = "val" if i < n_val else "train"
    print("split: " + ", ".join(f"circuit {g}: {sum(split[p] == 'val' for p, _ in v)} val / {len(v)}" for g, v in sorted(groups.items())))

    # labels are normalised to the original size, so a same-aspect resize needs no label changes
    src = {p: p for p, _ in matched}
    if a.max_side:
        cache = a.raw.resolve() / f"resized_{a.max_side}"
        with ThreadPoolExecutor() as ex:
            src = dict(zip(src, ex.map(lambda p: downscale(p, cache / p.parent.name / p.name, a.max_side), src)))
        print(f"images downscaled to max side {a.max_side}px in {cache}")

    out = a.out.resolve()
    for d in ("images", "labels"):
        shutil.rmtree(out / d, ignore_errors=True)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    n_box = Counter()
    for p, r in matched:
        W, H = r["size"] if r["size"] and all(r["size"]) else Image.open(p).size
        min_side = a.min_box_frac * max(W, H)
        lines = []
        for o in r["objects"]:
            if o["name"] not in cid:
                continue
            xs, ys = [q[0] for q in o["pts"]], [q[1] for q in o["pts"]]
            x0, x1, y0, y1 = max(0, min(xs)), min(W, max(xs)), max(0, min(ys)), min(H, max(ys))
            if x1 - x0 < min_side:
                c = (x0 + x1) / 2; x0, x1 = max(0, c - min_side / 2), min(W, c + min_side / 2)
            if y1 - y0 < min_side:
                c = (y0 + y1) / 2; y0, y1 = max(0, c - min_side / 2), min(H, c + min_side / 2)
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append(f"{cid[o['name']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            n_box[split[p]] += 1
        s = split[p]
        (out / "images" / s / p.name).symlink_to(src[p].resolve())
        (out / "labels" / s / p.with_suffix(".txt").name).write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_lines = [f"path: {json.dumps(str(out))}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    (out / "condition_attributes.json").write_text(json.dumps({
        "label_format": fmt, "merge_condition": a.merge_condition, "class_names": names,
        "class_counts": {n: counts[n] for n in names}, "dropped_classes": {n: counts[n] for n in dropped},
        "object_attributes": {c: {k: dict(v) for k, v in d.items()} for c, d in sorted(cond.items())},
        "image_attributes": {k: dict(v) for k, v in img_cond.items()},
    }, indent=2))
    n_tr = sum(v == "train" for v in split.values())
    print(f"wrote {out}/dataset.yaml: {len(names)} classes, train {n_tr} imgs/{n_box['train']} boxes, "
          f"val {len(split) - n_tr} imgs/{n_box['val']} boxes")


if __name__ == "__main__":
    main()
