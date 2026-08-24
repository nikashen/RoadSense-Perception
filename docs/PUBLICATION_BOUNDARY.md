# Publication Boundary

## Current state

The current `0.2.0.dev0` release candidate is an engineering foundation. Its 24-frame
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
dataset, a sequence-disjoint frozen split, a verified model/weight provenance
receipt bound into the evaluation report,
an independent evaluator, complete runtime evidence, and a sanitized report
whose hash binds all of the above. Failure or missing authorization must remain
visible instead of being converted into a positive result.

## BDD100K Detection lane

The first formal lane is intentionally narrower than the rest of the
multi-task roadmap: BDD100K Detection 2020 `val`, using the official devkit
commit `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`. The complete preparation,
inference, evaluator, and finalization contract is documented in
[BDD100K benchmark](BDD100K_BENCHMARK.md).

Until a real source receipt, frozen image manifest, model/inference receipts,
and two matching official evaluator receipts exist, the public state remains
`benchmark_claim_available=false`. The checked-in COCO YOLO11n adapter is an
out-of-domain baseline; it is not described as BDD-trained. A local `val`
receipt is not a hidden-test leaderboard result, and it does not authorize
claims about tracking or segmentation.
