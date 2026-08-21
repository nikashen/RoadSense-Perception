from __future__ import annotations

import numpy as np
import pytest

from roadsense.contracts import BoxXYXY, Detection
from roadsense.geometry import greedy_iou_match, intersection_over_union, pairwise_iou


def _detection(
    x_min: float,
    *,
    category_id: int = 1,
    score: float = 1.0,
) -> Detection:
    return Detection(
        category_id=category_id,
        score=score,
        bbox=BoxXYXY(x_min=x_min, y_min=0, x_max=x_min + 10, y_max=10),
    )


def test_iou_identity_and_disjoint() -> None:
    box = _detection(0).bbox
    assert intersection_over_union(box, box) == 1.0
    assert intersection_over_union(box, _detection(20).bbox) == 0.0


def test_iou_partial_overlap() -> None:
    assert intersection_over_union(_detection(0).bbox, _detection(5).bbox) == pytest.approx(1 / 3)


def test_pairwise_iou_shape_and_values() -> None:
    matrix = pairwise_iou((_detection(0), _detection(20)), (_detection(0),))
    np.testing.assert_allclose(matrix, [[1.0], [0.0]])


def test_greedy_match_is_class_aware() -> None:
    result = greedy_iou_match((_detection(0, category_id=1),), (_detection(0, category_id=2),))
    assert not result.matches
    assert result.unmatched_left == (0,)
    assert result.unmatched_right == (0,)


def test_greedy_match_returns_deterministic_pairs() -> None:
    left = (_detection(0), _detection(20))
    right = (_detection(1), _detection(19))
    result = greedy_iou_match(left, right, iou_threshold=0.5)
    assert [(match.left_index, match.right_index) for match in result.matches] == [(0, 0), (1, 1)]


def test_greedy_match_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        greedy_iou_match((), (), iou_threshold=1.1)
