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

`/api/v1/demo` returns `roadsense.demo/v1`, including a fixture ID, canvas,
ordered frames, objects in `xywh` display coordinates, segmentation polygons,
and evidence flags. `/api/v1/report` returns a sanitized fixture report with
the same explicit non-benchmark boundary.

The future local inference API will add bounded image/video/session inputs,
frame sequence numbers, content hashes, and model readiness. It will not
replace the fixture endpoint or silently turn a browser demo into a hosted
model service.
