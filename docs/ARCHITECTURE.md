# Architecture

## Runtime separation

RoadSense has two intentionally separate paths:

- **Pages / CI:** procedural fixture, no model import, no data download, no
  quality claim;
- **local real-data path (planned):** explicit dataset manifest, local model
  artifact, sequence-aware split, evaluator, and sanitized report.

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
- `api/`: transport and readiness endpoints.
- `web/`: visualization only; display threshold changes cannot mutate reports.

## Planned `v0.2`

The 3D milestone will add explicit coordinate frames, calibration hashes,
point-cloud limits, camera–LiDAR fusion, BEV outputs, and 3D track state. It
will not be smuggled into the `v0.1` 2D claims.
