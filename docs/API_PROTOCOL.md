# API Protocol

Current endpoints:

```text
GET /api/v1/health
GET /api/v1/readiness
GET /api/v1/demo
GET /api/v1/report
```

The local server also serves `/`, `/app.js`, `/styles.css`, and
`/favicon.svg`. The API routes are registered before the root static mount so
an asset request cannot shadow a versioned endpoint. Responses use strict
Pydantic models and reject malformed fixture payloads at the server boundary.

`/api/v1/demo` returns `roadsense.demo/v2`, including a fixture ID, canvas,
ordered frames, continuous scene `actors`, prediction `objects` in `xywh`
display coordinates, segmentation polygons, separate detection and segmentation
ontologies, and evidence flags. Actors drive the physical illustration while
objects drive boxes, confidence filtering, and track diagnostics, so deliberate
prediction errors do not make scene participants teleport or disappear.
The v1 fixture contract contains exactly 24 ordered frames at 100 ms cadence.
`/api/v1/report` returns a sanitized fixture report with
the same explicit non-benchmark boundary. Its 16-character `report_id` binds
the complete report, including detailed diagnostics, with canonical JSON.
`/api/v1/readiness` reports `service_mode=fixture_replay`; `ready` does not mean
that a real detector or segmenter model is loaded.

The future local inference API will add bounded image/video/session inputs,
frame sequence numbers, content hashes, and model readiness. It will not
replace the fixture endpoint or silently turn a browser demo into a hosted
model service.
