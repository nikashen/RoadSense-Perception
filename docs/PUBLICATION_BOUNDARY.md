# Publication Boundary

## Current state

The current release candidate is an engineering foundation. Its 24-frame
procedural city-loop is a fixture, not BDD100K, COCO, MOT, KITTI, or nuScenes.
`evaluation_authorized=false`, `frozen=false`, and
`benchmark_claim_available=false` are deliberate.

The fixture may support claims about:

- schema validation and coordinate contracts;
- deterministic replay and frame stepping;
- overlay composition and display filtering;
- tracker state transitions and report plumbing;
- API/Pages asset integrity.

It may not support claims about mAP, mIoU, HOTA, IDF1, FPS, latency,
generalization, safety, or production readiness.

## Real-data release blockers

Before a real benchmark can be published, the project needs a licensed local
dataset, a sequence-disjoint frozen split, a model/weight provenance record,
an independent evaluator, complete runtime evidence, and a sanitized report
whose hash binds all of the above. Failure or missing authorization must remain
visible instead of being converted into a positive result.

