# RoadSense-Perception

RoadSense-Perception is an evidence-first visual perception laboratory. The
`v0.2` capability preview defines a clean, testable contract for 2D object
detection, semantic segmentation, multi-object tracking, local-data evaluation,
model-artifact verification, and runtime audit records. A later milestone can
add point clouds, camera-LiDAR calibration, BEV features, and 3D tracking.

It is deliberately not presented as an autonomous-driving safety system. The
Pages demo uses a procedural city-loop fixture so that the interface,
frame state, overlays, and evidence boundary can be checked without silently
downloading images or model weights.

**Live demo:** <https://nikashen.github.io/RoadSense-Perception/>

## Current milestone: `0.2.0.dev0` capability preview

The repository currently contains:

- strict frozen contracts for image frames, boxes, detections, manifests,
  and evaluation reports;
- geometry-safe IoU and deterministic greedy association;
- compact detection AP, semantic IoU, and tracking MOTA/identity-F1 protocols;
- an explicit IoU tracker baseline with aging, score filtering, and reset;
- a 24-frame synthetic road sequence containing successful tracks, misses,
  false positives, and an identity-switch case;
- a Pages build that emits a hashed `demo.json` from the same payload as the
  local API, with an emergency fixture fallback only when that artifact is unavailable;
- FastAPI endpoints and a responsive Perception Workbench;
- a local-only sequence evaluator with split, alignment, mask-size, and
  fail-closed evidence checks;
- a hash-bound model-artifact manifest/adapter seam that never loads a model
  implicitly, plus a fixture runtime audit record with explicit non-benchmark
  boundaries;
- strict JSON, canonical hashes, atomic reports, package smoke tests, and
  Linux/Windows CI scaffolding.

The fixture report is useful for checking plumbing only. It is not COCO mAP,
BDD100K AP, official MOT metrics, an FPS claim, or evidence of model quality.

## 60-second demo

Windows PowerShell:

```powershell
Set-Location <WORKSPACE_ROOT>\项目十\RoadSense-Perception
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[serve]"
.\.venv\Scripts\roadsense.exe serve --port 8100
```

Open <http://127.0.0.1:8100/>. The page supports play/pause, frame stepping,
confidence display filtering, detection/segmentation/tracking layer toggles,
track selection, a frame inventory, and an evidence view.

For a dependency-light check:

```powershell
.\.venv\Scripts\python.exe -m roadsense smoke --json
.\.venv\Scripts\python.exe scripts\build_pages.py --output dist\pages
```

To produce an auditable fixture runtime record (diagnostic only, not a model
FPS benchmark):

```powershell
.\.venv\Scripts\python.exe -m roadsense benchmark --iterations 3 `
  --output reports/runtime_fixture_v1.json
```

See [Runtime audit](docs/RUNTIME_AUDIT.md) for the schema and evidence boundary.

For an operator-supplied local model artifact, validate its manifest first and
then (optionally) verify the checkpoint hash below an explicit root. Neither
command instantiates a model:

```powershell
.\.venv\Scripts\python.exe -m roadsense audit-artifact .\artifacts\manifest.json --json
.\.venv\Scripts\python.exe -m roadsense verify-artifact .\artifacts\manifest.json `
  --root .\artifacts --output .\reports\artifact-receipt.json
```

## Engineering shape

```text
strict contracts + JSON/hash guard
              |
              v
       frame / box geometry
              |
       detector output contract
              +--> semantic masks --> mIoU protocol
              +--> IoU tracker -----> MOTA / identity protocol
              +--> AP protocol ------> sanitized report
              |
       FastAPI local service / Pages fixture adapter
```

The detector and segmenter interfaces are intentionally not coupled to Torch,
OpenCV, or a particular vendor runtime. Future adapters may load a local
Torchvision/ONNX model only after an explicit artifact manifest has verified
the checkpoint hash, class ontology, preprocessing, and runtime versions.

## Real-data plan

BDD100K is the planned real-data source because its detection, semantic
segmentation, and tracking tasks share a road-scene domain. A real run must be
operator-initiated and locally licensed. The repository will keep only a
source receipt, exact archive/tree hashes, a sequence-aware split manifest,
configuration, and sanitized aggregate reports. It will not mirror BDD images,
videos, or labels.

The intended protocol is sequence/video-disjoint Train/Development/Final:

1. prepare and audit the local archive;
2. freeze the ontology, preprocessing, model artifact, and thresholds on
   Development;
3. run the designated Final split once without labels mounted in the runner;
4. let an independent evaluator produce AP/AP50/AP75, mIoU, MOTA, IDF1,
   identity switches, and runtime breakdowns;
5. publish only if licenses, hashes, split provenance, and evaluator receipts
   are complete.

Until that chain exists, `evaluation_authorized=false` and `frozen=false` are
the only valid public state.

### Formal BDD100K Detection lane

The repository now contains a separate, fail-closed runner for the official
BDD100K Detection 2020 `val` split. It freezes the image inventory, runs
inference without exposing labels, invokes the pinned official devkit in an
isolated interpreter, and requires two matching evaluator receipts before a
sanitized benchmark receipt can be built. The current adapter is explicitly a
COCO-pretrained YOLO11n cross-domain baseline; it does not claim BDD training
and it never fabricates the `rider` or `traffic sign` classes.

See [BDD100K benchmark](docs/BDD100K_BENCHMARK.md) for the exact commands and
licence boundary. No BDD images, labels, predictions, private weights, or
absolute paths are committed. Until a licensed operator supplies the official
data (including both published package checksums) and the finalizer succeeds,
this repository has no public BDD metric claim; the Pages workbench remains
the deterministic fixture. A stale Berkeley mirror response, Kaggle label
export, or development subset is a blocked input—not a benchmark result.

The repository now includes an explicit local-data entry point for that plan.
`roadsense evaluate-local` consumes a strict sequence-aware JSON spec and
operator-provided frame/mask files only; it never downloads data.  Missing,
overlapping, or malformed inputs fail closed.  See
[Local evaluation](docs/LOCAL_EVALUATION.md) for the format and a fixture
dry-run command.  A development report may omit model provenance for
exploration; an authorized or frozen report must include a locally verified
`model_artifact` manifest/root pair.

For a fully reproducible, operator-run smoke receipt using four real COCO8
validation images and a hash-verified YOLO11n ONNX artifact, see
[Real-data evaluation](docs/REAL_EVALUATION.md).  That receipt is explicitly
development evidence on a tiny subset, not an official COCO benchmark or a
production runtime claim.
The tracked [sanitized aggregate receipt](docs/REAL_EVALUATION_RECEIPT.json)
contains provenance and metrics only; raw images, labels, weights, and the
full local report bundle remain excluded from source control.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m mypy src\roadsense
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
node --check src\roadsense\web\app.js
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe scripts\verify_pages.py dist\pages
```

The local gate is expected to be green before a development snapshot is
published. CI reports the exact test count. The development extra includes
the Starlette-compatible `httpx2` client used by the API contract tests.

## Resume positioning

RoadSense-Perception is a multi-task 2D visual perception and video-tracking
platform, not a claim of complete autonomous-driving perception. It is meant
to complement ForgeSight-AD's industrial anomaly detection with ordinary
object-level geometry, class masks, temporal association, and deployment
contracts.

Suggested resume bullets (the COCO8 smoke numbers below remain development
evidence; an authorized benchmark needs a licensed, sequence-disjoint split):

- Designed a sequence-aware perception pipeline covering detection,
  semantic segmentation, and multi-object tracking with separate metric
  protocols and fail-closed evidence manifests.
- Implemented deterministic IoU association, track aging, identity-switch
  accounting, mask confusion matrices, and original-resolution coordinate
  contracts.
- Delivered a FastAPI/Pages Perception Workbench with frame stepping, overlay
  controls, track inspection, and an explicit separation between fixture UI
  evidence and benchmark claims.
- Built a local-only, sequence-aware evaluation lane with hash-bound reports,
  plus a model-artifact manifest/verification receipt and framework-neutral
  adapter seam; authorized or frozen reports cannot omit model provenance.
- Built a reproducible COCO8 + YOLO11n ONNX development runner with frozen
  letterbox/post-processing settings and CPU runtime metadata; the four-image
  compact AP@0.50 is recorded as evidence, not official COCO mAP.

## Boundaries

See [Architecture](docs/ARCHITECTURE.md), [Data Protocol](docs/DATA_PROTOCOL.md),
[Evaluation Protocol](docs/EVALUATION_PROTOCOL.md),
[Tracking Protocol](docs/TRACKING_PROTOCOL.md),
[Model Artifact Protocol](docs/MODEL_ARTIFACT_PROTOCOL.md),
[API Protocol](docs/API_PROTOCOL.md), [Security](docs/SECURITY.md),
[Publication Boundary](docs/PUBLICATION_BOUNDARY.md),
[Publishing](docs/PUBLISHING.md), the [BDD100K benchmark](docs/BDD100K_BENCHMARK.md),
and the Chinese [resume case](docs/RESUME_PROJECT_ZH.md).
