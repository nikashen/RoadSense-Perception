# Tracking Protocol

RoadSense separates detection output, temporal association, and tracking
evaluation. The tracker consumes one ordered frame at a time and has no access
to labels, video decoding, or model weights.

## Baseline behavior

`roadsense.tracking-iou/v1` performs deterministic greedy IoU matching. Matches
are sorted by descending IoU and then by stable track/detection identifiers.
Unmatched detections start new tracks; unmatched tracks age until `max_age` and
are then retired. `reset()` must be called at every sequence boundary.

The Pages fixture intentionally includes a pedestrian disappearance and
reappearance so track aging and an identity-switch path are visible. This is a
contract test, not evidence of tracker quality.

## Real-data requirements

- Split by complete video/sequence, never by adjacent frames.
- Preserve source frame order and timestamps; reject duplicates and regressions.
- Freeze detector score threshold, association threshold, `max_age`, and class
  policy on Development before running Final.
- Report MOTA, IDF1, HOTA, identity switches, and fragmentations separately.
- Pin the evaluator version and record its input/output hashes.

The local identity-F1 helper is deliberately named and documented as a compact
diagnostic. It must not be presented as the official TrackEval implementation.
