# RoadSense-Perception

RoadSense-Perception is an evidence-first visual perception laboratory. The
first milestone defines a clean, testable contract for 2D object detection,
semantic segmentation, and multi-object tracking; a later `v0.2` milestone
will add point clouds, camera–LiDAR calibration, BEV features, and 3D tracking.

It is deliberately not presented as an autonomous-driving safety system. The
public Pages demo uses a procedural city-loop fixture so that the interface,
frame state, overlays, and evidence boundary can be checked without silently
downloading images or model weights.

## Current milestone: `0.1.0.dev0`

The repository currently contains:

- strict immutable contracts for image frames, boxes, detections, manifests,
  and evaluation reports;
- geometry-safe IoU and deterministic greedy association;
- compact detection AP, semantic IoU, and tracking MOTA/identity-F1 protocols;
- an explicit IoU tracker baseline with aging, score filtering, and reset;
- a 24-frame synthetic road sequence containing successful tracks, misses,
  false positives, and an identity-switch case;
- FastAPI endpoints and a responsive Perception Workbench;
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
\.venv\Scripts\python.exe -m roadsense smoke --json
\.venv\Scripts\python.exe scripts\build_pages.py --output dist\pages
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

BDD100K is the planned `v0.1` real-data source because its detection, semantic
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

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m mypy src\roadsense
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
node --check src\roadsense\web\app.js
.\.venv\Scripts\python.exe -m build
```

The initial local gate has 56 passing tests. A Starlette/httpx deprecation
warning may appear in the optional FastAPI test client; it does not change the
fixture or API result.

## Resume positioning

RoadSense-Perception is a multi-task 2D visual perception and video-tracking
platform, not a claim of complete autonomous-driving perception. It is meant
to complement ForgeSight-AD's industrial anomaly detection with ordinary
object-level geometry, class masks, temporal association, and deployment
contracts.

Suggested resume bullets after the real-data milestone is authorized:

- Designed a sequence-aware perception pipeline covering detection,
  semantic segmentation, and multi-object tracking with separate metric
  protocols and fail-closed evidence manifests.
- Implemented deterministic IoU association, track aging, identity-switch
  accounting, mask confusion matrices, and original-resolution coordinate
  contracts.
- Delivered a FastAPI/Pages Perception Workbench with frame stepping, overlay
  controls, track inspection, and an explicit separation between fixture UI
  evidence and benchmark claims.

## Boundaries

See [Architecture](docs/ARCHITECTURE.md), [Data Protocol](docs/DATA_PROTOCOL.md),
[Evaluation Protocol](docs/EVALUATION_PROTOCOL.md),
[Tracking Protocol](docs/TRACKING_PROTOCOL.md),
[Model Artifact Protocol](docs/MODEL_ARTIFACT_PROTOCOL.md),
[API Protocol](docs/API_PROTOCOL.md), [Security](docs/SECURITY.md),
[Publication Boundary](docs/PUBLICATION_BOUNDARY.md),
[Publishing](docs/PUBLISHING.md), and the Chinese [resume case](docs/RESUME_PROJECT_ZH.md).
