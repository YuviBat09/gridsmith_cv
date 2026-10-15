#!/usr/bin/env python
"""Generate a tiny fake EPRI dataset mirroring the real Kaggle label format.

Creates <out>/raw/Circuit5.zip, <out>/raw/Circuit10.zip (nested folders, random JPGs named "<circuit> (n).JPG")
and <out>/Overhead-Distribution-Labels.csv with columns "Label","External ID", where Label is a Python-repr dict
(like the real file) with 'polygon' and 'line' objects and 'obscured/truncated' classifications. Also includes
labels for a circuit that is not present locally (to exercise the unmatched path).

  python src/tests/make_fake_dataset.py --out /path/to/fake
"""
import argparse, csv, io, random, shutil, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["insulator", "conductor", "pole", "crossarm", "cutouts", "transformers", "guy_wires"]
W, H = 640, 480


def fake_label(rng):
    objs = []
    for _ in range(rng.randint(2, 8)):
        cls = rng.choice(CLASSES)
        if cls in ("conductor", "guy_wires"):  # wires are polylines in the real data, often edge-to-edge
            x = rng.uniform(0, W)
            o = {"value": cls, "line": [{"x": x, "y": 0}, {"x": x + rng.uniform(-20, 20), "y": H}], "instanceURI": "None"}
        else:
            cx, cy, r = rng.uniform(50, W - 50), rng.uniform(50, H - 50), rng.uniform(10, 60)
            o = {"value": cls, "polygon": [{"x": float(cx + r * np.cos(t)), "y": float(cy + r * np.sin(t))} for t in np.linspace(0, 6, 6)],
                 "instanceURI": "None"}
        if rng.random() < 0.3:
            o["classifications"] = [{"value": rng.choice(["obscured/truncated", "obscured_truncated"]),
                                     "answer": {"value": rng.choice(["obscured", "truncated", "both"])}}]
        objs.append(o)
    img_cls = [{"value": "image_quality", "answer": {"value": "motion_blur"}}] if rng.random() < 0.2 else []
    return {"objects": objs, "classifications": img_cls, "relationships": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="images per circuit")
    a = ap.parse_args()
    rng = random.Random(0)
    shutil.rmtree(a.out, ignore_errors=True)
    (a.out / "raw").mkdir(parents=True)
    rows = []
    for circuit in (5, 10):
        with zipfile.ZipFile(a.out / "raw" / f"Circuit{circuit}.zip", "w") as zf:
            for i in range(1, a.n + 1):
                buf = io.BytesIO()
                Image.fromarray(np.random.default_rng(i).integers(0, 255, (H, W, 3), dtype=np.uint8)).save(buf, "JPEG")
                zf.writestr(f"Circuit{circuit}/sub dir/{circuit} ({i}).JPG", buf.getvalue())  # nested folder
                if i != a.n:  # last image unlabeled
                    ext = f"{circuit}%20({i}).jpg" if i == 2 else f"{circuit} ({i}).JPG"  # URL-encoded / lowercase variant
                    rows.append({"Label": repr(fake_label(rng)), "External ID": ext})
            zf.writestr(f"__MACOSX/Circuit{circuit}/._{circuit} (1).JPG", b"junk")
    rows += [{"Label": repr(fake_label(rng)), "External ID": f"1 ({i}).JPG"} for i in range(1, 4)]  # not local
    with open(a.out / "Overhead-Distribution-Labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Label", "External ID"], quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)
    print(f"fake dataset at {a.out}: {len(rows)} label rows")


if __name__ == "__main__":
    main()
