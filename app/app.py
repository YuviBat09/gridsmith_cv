#!/usr/bin/env python
"""Gradio MVP for EPRI distribution-pole inspection.

Run:  python app/app.py   ->  http://127.0.0.1:7860
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from infer import (conclude, draw, load_model, model_info, predict, summarize,  # noqa: E402
                   write_reports)

MODEL = load_model(os.environ.get("EPRI_WEIGHTS"))
INFO = model_info(MODEL)

STATUS_COLORS = {"OK": "#1a7f37", "Review": "#b58100", "Action needed": "#c62828"}
DISPLAY_MAX = 2048  # longest side of gallery previews, px
TABLE_COLS = ["image", "status", "detections", "inventory", "issues", "low-conf"]


def banner_html() -> str:
    if INFO.get("demo"):
        bg, fg = "#fff4e5", "#8a4b00"
    else:
        bg, fg = "#e8f5e9", "#1b5e20"
    return (f"<div style='padding:10px 14px;border-radius:8px;background:{bg};color:{fg};"
            f"font-weight:600'>{html.escape(INFO['banner'])}</div>")


def summary_html(summary: dict | None) -> str:
    if not summary:
        return "<div style='color:#666'>Upload images and press <b>Inspect</b>.</div>"
    color = STATUS_COLORS[summary["status"]]
    items = "".join(f"<li>{html.escape(b)}</li>" for b in summary["bullets"])
    return (f"<div><span style='display:inline-block;padding:6px 14px;border-radius:999px;"
            f"background:{color};color:white;font-weight:700;font-size:1.1em'>"
            f"Overall: {summary['status']}</span><ul style='margin-top:10px'>{items}</ul></div>")


def _path_of(f) -> str:
    return f if isinstance(f, str) else getattr(f, "name", None) or f["name"]


def inspect(files, conf, tile, imgsz):
    if not files:
        return [], pd.DataFrame(columns=TABLE_COLS), summary_html(None), None
    out_dir = Path(tempfile.mkdtemp(prefix="epri_inspect_"))
    (out_dir / "images").mkdir()
    gallery, rows, per_image, full_res = [], [], [], []
    for f in files:
        p = Path(_path_of(f))
        img = cv2.imread(str(p))
        if img is None:
            rows.append([p.name, "unreadable", 0, "", "could not read image", 0])
            continue
        dets = predict(MODEL, img, conf=float(conf), imgsz=int(imgsz), tile=bool(tile))
        fnd = conclude(dets, MODEL.names)
        ann = draw(img, dets)
        ann_path = out_dir / "images" / f"{p.stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), ann)
        # Downscaled copy for the gallery so the whole photo fits on screen.
        scale = DISPLAY_MAX / max(ann.shape[:2])
        if scale < 1:
            ann = cv2.resize(ann, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        view_path = out_dir / f"{p.stem}_view.jpg"
        cv2.imwrite(str(view_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
        full_res.append(str(ann_path))
        gallery.append((str(view_path), f"{p.name} - {fnd['status']}"))
        per_image.append({"image": p.name, "detections": dets, "findings": fnd})
        rows.append([p.name, fnd["status"], fnd["n_detections"],
                     ", ".join(f"{c}: {n}" for c, n in fnd["inventory"].items()),
                     " ".join(i["message"] for i in fnd["issues"]) or "-",
                     len(fnd["low_confidence"])])
    if not per_image:
        return gallery, pd.DataFrame(rows, columns=TABLE_COLS), summary_html(None), None
    j, c, d = write_reports(per_image, out_dir, INFO)
    return (gallery, pd.DataFrame(rows, columns=TABLE_COLS),
            summary_html(summarize(per_image)), [str(c), str(j), str(d)] + full_res)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EPRI Pole Inspection") as demo:
        gr.Markdown("## Distribution Pole Inspection\nUpload drone photos of overhead "
                    "distribution poles to detect components and flag issues for review.")
        gr.HTML(banner_html())
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Images", file_count="multiple", file_types=["image"])
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                imgsz = gr.Dropdown([640, 1024, 1280], value=1024, label="Inference size (px)")
                tile = gr.Checkbox(False, label="Tiled inference (better for small parts in "
                                                "large images; slower)")
                btn = gr.Button("Inspect", variant="primary")
                summary = gr.HTML(summary_html(None))
                report = gr.File(label="Downloads: report (CSV / JSON) + full-res annotated images",
                                 interactive=False)
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Annotated images (click to enlarge; full-res in downloads)",
                                     columns=2, height=600, object_fit="contain",
                                     buttons=["download", "fullscreen"])
                table = gr.Dataframe(headers=TABLE_COLS, label="Per-image findings",
                                     wrap=True, interactive=False)
        btn.click(inspect, [files, conf, tile, imgsz], [gallery, table, summary, report],
                  api_name="inspect")
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                      server_port=int(os.environ.get("PORT", 7860)))
