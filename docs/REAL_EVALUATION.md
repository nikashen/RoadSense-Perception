# Real-data smoke evaluation (COCO8 + YOLO11n)

This document records the first real-data run for the v0.2 capability lane. It
is deliberately a development receipt, not a benchmark publication. The run
uses the eight-image Ultralytics COCO8 package (four validation images) and a
local Ultralytics YOLO11n ONNX export. The image and weight files stay under
the ignored `data/raw/` directory and are not committed to Git, wheels, sdist,
Pages, or reports published from this repository.

## Inputs and provenance

| Input | Upstream record | Local receipt |
| --- | --- | --- |
| COCO8 package | [Ultralytics COCO8 asset](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip); the package's README identifies four train and four validation images | `data/raw/real_eval/coco8.zip`, 443,158 bytes, SHA-256 `54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9` |
| YOLO11n ONNX | [Ultralytics assets release v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx); the file metadata reports Ultralytics export `8.3.237`, dynamic input, and `opset=22` | `data/raw/real_eval/yolo11n.onnx`, 10,930,182 bytes, SHA-256 `634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b` |

The dataset page/revision and archive hash are the authoritative dataset
identity. Release URLs can redirect or become temporarily unavailable; a
different archive must never be substituted without changing the receipt and
rerunning the evaluation. The model export metadata declares 80 COCO classes,
input size 640 x 640, raw output `output0` with shape `(1, 84, anchors)`, and
`nms=false`.

### License boundary

The downloaded package includes a GPL-3.0 license file for the Ultralytics
package. The model metadata identifies the YOLO11n export as AGPL-3.0. Those
notices do **not** by themselves settle the license or redistribution terms of
the underlying COCO images and annotations. Treat the local copy as
operator-provided, private evaluation input; perform an independent current
license review before sharing any image, label, model, or derived artifact.

## Isolated runtime

The working project environment is intentionally dependency-light. Inference
was run in a separate environment to avoid importing optional vision packages
from the Pages/API path:

```text
CPython 3.10.2 (standard Windows build)
onnxruntime 1.23.2
numpy 1.26.4
Pillow 12.3.0
CPUExecutionProvider
```

The original Anaconda-backed project environment had ONNX Runtime 1.18.1,
which rejects the model's opset 22. Newer wheels also hit a Windows DLL
initialisation failure in that Anaconda environment, so the isolated
`.venv-ort` runtime is part of this receipt. It is not a new mandatory
dependency of the core package. The package's optional `vision` extra therefore
requires ONNX Runtime 1.23 or newer; older runtimes remain unsuitable for this
YOLO11n artifact even though the framework-neutral artifact contract accepts
other runtime versions for other models.

## Reproduction command

After placing the locally licensed files at the paths above, run the dedicated
runner with the isolated interpreter:

```powershell
py -3.10 -m venv .venv-ort
.\.venv-ort\Scripts\python.exe -m pip install `
  "onnxruntime==1.23.2" "numpy==1.26.4" "Pillow==12.3.0" "pydantic>=2.8,<3"

# The isolated environment does not need the package's optional dependencies.
# Point it at the checked-out source (or install the project with --no-deps).
$env:PYTHONPATH = "$PWD\src"

.\.venv-ort\Scripts\python.exe scripts\run_coco8_onnx_eval.py `
  --dataset-root data\raw\real_eval\coco8 `
  --model data\raw\real_eval\yolo11n.onnx `
  --archive data\raw\real_eval\coco8.zip `
  --output-dir reports\coco8_real `
  --split val `
  --score-threshold 0.25 `
  --nms-iou 0.70
```

The runner converts YOLO normalized `xywh` labels to the RoadSense original
pixel `xyxy` contract, records an input/model verification receipt, and calls
`roadsense evaluate-local`. It has no network code; downloading and license
review happen before the command.

The repository tracks a [sanitized aggregate receipt](REAL_EVALUATION_RECEIPT.json)
with the same hashes, protocol settings, and metrics. It intentionally omits
raw images, labels, model weights, absolute paths, per-detection predictions,
and machine-specific timing; the complete JSON bundle remains local and
ignored by Git.

## Frozen preprocessing and postprocessing settings

- RGB conversion; aspect-preserving letterbox to 640 x 640.
- Padding value 114 (image-space value), then float32 NCHW normalization by
  `1/255`.
- Raw YOLO output is decoded as center `xywh` plus 80 class scores.
- Per-class greedy NMS, IoU threshold `0.70`.
- Detection score threshold `0.25`; maximum detections per image `300`.
- Boxes are mapped back to each source image's original width and height and
  clipped before constructing `FrameRecord` values.
- Each validation image is represented as one single-frame sequence. Therefore
  this run exercises detection and report provenance only; it is not a video
  tracking evaluation.

## Observed result

The four-image validation split contains 17 ground-truth objects. With the
RoadSense compact `roadsense.detection-ap/v1` protocol at IoU `0.50`, the run
produced:

| Metric | Value |
| --- | ---: |
| compact AP (macro over present classes) | 0.650000 |
| precision | 0.6923076923 |
| recall | 0.5294117647 |
| true positives | 9 |
| false positives | 4 |
| false negatives | 8 |

For the recorded `reports/coco8_real` layout, the generated local report had
the content-bound `report_id=8163e7d374440aa2` and bound the verified model
artifact receipt. Moving the output directory changes the relative model,
spec, and dependency-lock paths included in the bound content and therefore
produces a new report ID. The separate `run_id` includes machine timing and
may change on a rerun. Its evidence flags were:

```text
evidence_level=development
evaluation_authorized=false
frozen=false
benchmark_claim_available=false
```

These values are **not** COCO mAP, AP50/AP75, official Ultralytics metrics,
BDD100K metrics, or a generalized model-quality claim. Four images are a tiny
smoke subset and are insufficient for model selection, safety conclusions, or
production readiness. Any timing emitted by the runner is a local diagnostic;
it must not be presented as model FPS or a serving SLA.

## Publication rule

Keep the raw archive, images, labels, ONNX file, and private absolute paths
outside source control. A future public benchmark would additionally require
a licensed and sequence-disjoint dataset manifest, an independently pinned
evaluator, a frozen model/preprocessing receipt, complete runtime evidence, and
explicit authorization. Until then, retain the development boundary above.
