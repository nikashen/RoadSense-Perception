# Evaluation Protocol

The fixture protocols are intentionally compact and named so they cannot be
mistaken for third-party leaderboard implementations.

## Detection

`roadsense.detection-ap/v1` sorts predictions by score (with a canonical
frame/geometry tie-break), matches class-aware boxes at a configured IoU
threshold, and reports AP, precision, recall, and per-class counts. Ground
truth boxes are canonically ordered by geometry before matching, so equivalent
input permutations cannot change the result. The top-level `ap` is the
unweighted mean of per-class AP over classes with at least one reference object.
It is not COCO mAP and does not implement the full COCO area/size/max-detection
matrix.

## Segmentation

`roadsense.semantic-iou/v1` computes a confusion matrix, pixel accuracy,
per-class IoU, and mean IoU after removing an explicit ignore index. Labels
outside the ontology are rejected. The dense confusion matrix is bounded to
four million cells; callers must use a smaller class ontology or a sparse,
separately specified evaluator for larger label spaces.

The fixture segmentation ontology is `0=background`, `1=road`, `2=car`,
`3=vulnerable road user`, and `4=sidewalk`; the display polygons and raster
mask use the same IDs.

## Tracking

`roadsense.tracking-iou/v1` uses framewise IoU association, counts false
positives/negatives and identity switches, and reports MOTA plus a
maximum-count identity F1. The identity F1 is not the TrackEval reference
implementation; a sequence with no reference detections is rejected because
MOTA has no defined denominator. Track IDs are global within each sequence and
must retain one category; cross-frame category changes are rejected instead of
silently merging identities. A real BDD/MOT report must use the designated
official or independently pinned evaluator.

An all-empty ground-truth sequence is rejected. MOTA and identity scores are
undefined in that case and must not be rendered as a misleading perfect score.

## Aggregation

Detection `ap` is the macro mean of per-class AP over classes with at least one
ground-truth instance, at the configured IoU threshold. It is not a pooled COCO
mAP value. Segmentation `mean_iou` likewise averages only classes present in
the evaluated pixels.

## Runtime

Cold start, preprocessing, inference, association, rendering, device,
package versions, and artifact hashes must be recorded separately. A browser
display threshold is never an evaluation threshold.
