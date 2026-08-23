# Architecture

## Runtime separation

RoadSense has two intentionally separate paths:

- **Pages / CI:** procedural fixture, no model import, no data download, no
  quality claim;
- **local real-data path:** explicit dataset manifest, local sequence-aware
  input spec, evaluator, and sanitized report.  Model adapters remain
  separately opt-in and must provide their own artifact provenance.

The API does not accept a checkpoint path from a browser request. A future
server configuration will select a named local adapter and verify its artifact
before the first frame is processed.

## Module boundaries

- `contracts.py`: frozen top-level validation models; no FastAPI or Torch dependency.
- `geometry.py`: box geometry and association primitives.
- `metrics/`: pure detection, segmentation, and tracking evaluators.
- `tracking/`: consumes detections and frame order only; it never opens media.
- `fixture.py`: deterministic synthetic road scene and adapter payload.
- `evidence.py` / `json_io.py`: publication gates, canonical JSON, hashes, and
  atomic writes.
- `adapters.py`: dependency-light model-artifact manifests, local SHA-256
  receipts, and an explicit adapter registry; it never imports a model runtime.
- `api/`: transport and readiness endpoints.
- `web/`: visualization only; display threshold changes cannot mutate reports.

## Current `v0.2` capability preview

The local evaluation, model-artifact, and runtime-audit lanes are now wired as
explicit operator commands. They remain opt-in and local-only: the Pages/API
fixture path does not import an inference runtime, fetch data, or claim model
quality. A verified model artifact is required before an authorized/frozen
local report can be produced.

## Planned `v0.3`

The 3D milestone will add explicit coordinate frames, calibration hashes,
point-cloud limits, camera–LiDAR fusion, BEV outputs, and 3D track state. It
will not be smuggled into the current 2D claims.
