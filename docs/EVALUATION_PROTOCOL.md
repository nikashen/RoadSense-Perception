# Evaluation Protocol

The fixture protocols are intentionally compact and named so they cannot be
mistaken for third-party leaderboard implementations.

## Detection

`roadsense.detection-ap/v1` sorts predictions by score, matches class-aware
boxes at a configured IoU threshold, and reports AP, precision, recall, and
per-class counts. It is not COCO mAP and does not implement the full COCO
area/size/max-detection matrix.

## Segmentation

`roadsense.semantic-iou/v1` computes a confusion matrix, pixel accuracy,
per-class IoU, and mean IoU after removing an explicit ignore index. Labels
outside the ontology are rejected.

## Tracking

`roadsense.tracking-iou/v1` uses framewise IoU association, counts false
positives/negatives and identity switches, and reports MOTA plus a
maximum-count identity F1. The identity F1 is not the TrackEval reference
implementation; a real BDD/MOT report must use the designated official or
independently pinned evaluator.

## Runtime

Cold start, preprocessing, inference, association, rendering, device,
package versions, and artifact hashes must be recorded separately. A browser
display threshold is never an evaluation threshold.

