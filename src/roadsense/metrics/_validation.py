"""Validation shared by sequence metrics.

The public metric functions intentionally operate on frame records rather than
on an implicit positional array.  Before calculating a score we therefore
validate the sequence identity explicitly.  This prevents duplicate frame
indices from silently overwriting entries in the detection evaluator and
prevents timestamp regressions from producing misleading tracking reports.
"""

from __future__ import annotations

from itertools import pairwise

from roadsense.contracts import FrameRecord


def validate_aligned_sequences(
    ground_truth: tuple[FrameRecord, ...],
    predictions: tuple[FrameRecord, ...],
) -> None:
    """Validate non-empty, frame-aligned sequences used by a metric.

    Frame indices need not be contiguous (a caller may intentionally evaluate
    a sampled sequence), but they must be unique and strictly increasing in
    both inputs.  Timestamps may repeat for sources with coarse clocks, yet a
    regression is always rejected.  The prediction and reference sequence are
    compared by frame index; image dimensions are deliberately not required to
    be equal because adapters may emit boxes in a declared, common coordinate
    space while carrying a resized image descriptor.
    """

    if len(ground_truth) != len(predictions) or not ground_truth:
        raise ValueError("ground-truth and prediction sequences must be non-empty and aligned")

    truth_indices = tuple(frame.frame_index for frame in ground_truth)
    prediction_indices = tuple(frame.frame_index for frame in predictions)
    if truth_indices != prediction_indices:
        raise ValueError("frame indices must align")

    for frames, role in ((ground_truth, "ground-truth"), (predictions, "prediction")):
        for previous, current in pairwise(frames):
            if current.frame_index <= previous.frame_index:
                raise ValueError(f"{role} frame indices must be unique and strictly increasing")
            if current.timestamp_ms < previous.timestamp_ms:
                raise ValueError(f"{role} timestamps must be monotonic")
