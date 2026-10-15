# EPRI Distribution Inspection: CV MVP

This project trains a YOLO detector on the EPRI Distribution Inspection Imagery dataset, which contains
drone photos of overhead distribution poles. It also includes a small web app that shows each detected
component and flags possible problems.

## Quick start (just run the app)

macOS / Linux, Python 3.10+:

```bash
git clone https://github.com/YuviBat09/gridsmith_cv.git && cd gridsmith_cv && ./run.sh
```

The first run creates `.venv`, installs dependencies (a few minutes), downloads the trained model from the
[latest release](https://github.com/YuviBat09/gridsmith_cv/releases/latest), and opens
http://127.0.0.1:7860. Later runs start in seconds. Use `PORT=8080 ./run.sh` to change the port. No dataset
download is needed to use the app; that is only for retraining (below).

## Project layout

```
data/raw/            Circuit5.zip, Circuit10.zip, labels file (you download these)
data/yolo/           YOLO-format dataset produced by src/prepare_data.py
models/best.pt       trained weights (src/train.py)
models/metrics.json  validation metrics (mAP, per-class AP, class names)
src/infer.py         inference library + batch CLI
app/app.py           Gradio web app
```

## 1. Setup

```bash
cd epri-cv
python3 -m venv .venv
source .venv/bin/activate
pip install torch ultralytics opencv-python-headless gradio pandas
```

## 2. Download the data (manual steps)

1. **Images, from IEEE DataPort.** You need a free IEEE account. Log in, open the *EPRI Distribution
   Inspection Imagery* dataset page, and download **Circuit5.zip** and **Circuit10.zip**. Put both files in
   `data/raw/`. You don't need to unzip them yourself, because the prepare script handles that.
2. **Labels, from Kaggle.** Download the annotations from
   [kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery).
   You can use the website, or the CLI:
   `kaggle datasets download dexterlewis/epri-distribution-inspection-imagery -p data/raw --unzip`.
   Then note the path of the labels file.

## 3. Prepare, train, run

```bash
# convert the images and labels to YOLO format (wire classes are drawn as lines, so we drop them;
# images are downscaled to 2048px once into data/raw/resized_2048, labels are unaffected)
python src/prepare_data.py --raw data/raw --labels data/raw/Overhead-Distribution-Labels.csv \
    --out data/yolo --drop-classes conductor,other_wire,background_structure

# smoke test (a few minutes) to confirm the pipeline works
python src/train.py --data data/yolo/dataset.yaml --quick

# full training run; writes models/best.pt and models/metrics.json
python src/train.py --data data/yolo/dataset.yaml

# batch inference: annotated images + findings.json + findings.csv + detections.csv
python src/infer.py path/to/img_or_folder --out runs/infer [--conf 0.25] [--tile]

# web app
python app/app.py        # then open http://127.0.0.1:7860
```

If `models/best.pt` doesn't exist yet, both the CLI and the app run in **DEMO MODE**. They use a
COCO-pretrained `yolo11n.pt`, which downloads automatically, so you can try the UI before training.
In this mode the classes are generic objects such as person or bus, not utility components. A banner in
the app shows which model is loaded. For a trained model, the banner also shows mAP from
`models/metrics.json`.

To use different weights, pass `python src/infer.py ... --weights path.pt` or set
`EPRI_WEIGHTS=path.pt python app/app.py`.

## Current results

YOLO26s fine-tuned on Circuit 5 (338 train / 84 val images, 50 epochs at 1024px, ~80 min on an Apple M5):
mAP50 **0.90** (pole 0.94, transformers 0.93, crossarm 0.89, insulator 0.88, cutouts 0.86). Full numbers are
in `models/metrics.json`. Weights are not committed; retrain with the commands above.

Notes from testing:
- The EPRI labels contain **component locations only**, with no damage or condition classes. The only
  per-object attribute is `obscured/truncated`. Defect conclusions need extra labeling.
- YOLO26 is NMS-free, so `infer.py` removes duplicate same-class boxes itself (IoU > 0.5 or 90% nested).
- Images under 2500px (screenshots, crops) are run at 320 + 640px and merged. Running close-ups at 1024px
  upscales parts far beyond the training scale, and the model misses them.
- Keep the confidence at 0.25. On held-out images, higher thresholds made per-image counts worse.

## How findings are produced

`src/infer.py` turns detections into findings using rules. All of the rules are in the `RULES` dict at the
top of that file:

- **Inventory**: the number of objects detected in each class.
- **Defects**: any class name (or condition attribute) that contains keywords such as `broken`, `damage`,
  `crack`, `burn`, `missing`, `contact`, `rust` or `leaning`. Each keyword has a severity.
- **Vegetation encroachment**: a vegetation box that overlaps a conductor, pole, crossarm or transformer
  box. Overlap is measured as intersection divided by the area of the smaller box.
- **Low-confidence detections**: anything below `low_conf_threshold` (default 0.40) is listed for a human
  to review.
- **Status**:
  - `Action needed` if any high-severity issue was found.
  - `Review` if there are only medium-severity issues or low-confidence detections.
  - `OK` otherwise.

**Tiled inference** (`--tile` in the CLI, or the checkbox in the app) is a simple SAHI-style approach with
no extra dependencies. It predicts on overlapping tiles plus the full image, then merges the results with
class-wise NMS. It helps with small hardware, such as cutouts, arresters and fuses, in high-resolution
drone images. It is slower: roughly the number of tiles + 1 forward passes per image.

## Limitations

- These are **decision-support results, not an inspection verdict.** Every flagged item, and every
  "OK", should be checked by a qualified person.
- Defect detection only works if the training labels encode condition, either as separate classes such
  as `insulator_broken` or as attributes. If the trained model only knows component classes, the app can
  only build inventories and flag vegetation overlap.
- Vegetation encroachment comes from 2D box overlap, so it has no sense of depth. A tree *behind* a line
  is flagged in the same way as one touching it.
- The dataset covers only two circuits (Circuit5 and Circuit10), so performance on other utilities,
  regions, seasons or camera setups may be much lower.
- Box-level detection can't measure clearances, lean angles or the extent of corrosion.
- In demo mode the findings are meaningless apart from showing that the pipeline works.
- The app keeps report files in a temp directory and runs single-user and in-process, with no auth.
  It isn't meant to be exposed publicly as-is.

## Licensing

- Ultralytics YOLO is **AGPL-3.0**. Commercial or hosted use requires either releasing the full app
  source or an Ultralytics Enterprise license.
- EPRI labels (Kaggle) are CC BY-SA 4.0. EPRI images (IEEE DataPort, DOI 10.21227/rh5n-xn68) are not
  redistributed here. Check their terms before commercial use.
