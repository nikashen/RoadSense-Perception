from __future__ import annotations

from itertools import pairwise

import pytest

from roadsense.contracts import EvaluationReport, EvidenceLevel
from roadsense.evidence import (
    assert_publication_authorized,
    build_fixture_report,
    compute_report_id,
)
from roadsense.fixture import (
    FRAME_COUNT,
    VEHICLE_TRACK_IDS,
    build_demo_payload,
    build_fixture_bundle,
    build_fixture_metrics,
)


def test_fixture_is_deterministic() -> None:
    first = build_demo_payload()
    second = build_demo_payload()
    assert first == second
    assert len(first["frames"]) == FRAME_COUNT


def test_fixture_contains_all_three_tasks() -> None:
    payload = build_demo_payload()
    assert set(payload["metrics"]) == {"detection", "segmentation", "tracking"}
    assert payload["evidence"]["benchmark_claim_available"] is False  # type: ignore[index]


def test_fixture_bundle_shapes() -> None:
    bundle = build_fixture_bundle()
    assert len(bundle.truth_frames) == FRAME_COUNT
    assert bundle.truth_masks.shape == (FRAME_COUNT, 90, 160)
    assert bundle.truth_masks.shape == bundle.prediction_masks.shape


def test_fixture_vehicles_follow_perspective_lanes_instead_of_sliding_sideways() -> None:
    bundle = build_fixture_bundle()
    vehicle_tracks: dict[int, list[tuple[float, float, float, float]]] = {}
    for frame in bundle.truth_frames:
        for item in frame.detections:
            if item.track_id in VEHICLE_TRACK_IDS:
                vehicle_tracks.setdefault(item.track_id, []).append(
                    (item.bbox.x_min, item.bbox.y_min, item.bbox.width, item.bbox.height)
                )

    assert set(vehicle_tracks) == set(VEHICLE_TRACK_IDS)
    for boxes in vehicle_tracks.values():
        assert len(boxes) >= 19
        bottom_centers = [y + height for _x, y, _width, height in boxes]
        widths = [width for _x, _y, width, _height in boxes]
        centers_x = [x + width / 2.0 for x, _y, width, _height in boxes]
        assert all(next_value > value for value, next_value in pairwise(bottom_centers))
        assert all(next_value > value for value, next_value in pairwise(widths))
        assert abs(centers_x[-1] - centers_x[0]) < bottom_centers[-1] - bottom_centers[0]


def test_fixture_segmentation_ontology_matches_raster_classes() -> None:
    report = build_fixture_metrics()["segmentation"]
    assert len(report["per_class_iou"]) == 5  # type: ignore[index]
    assert report["evaluated_pixels"] == FRAME_COUNT * 90 * 160  # type: ignore[index]


def test_fixture_report_has_stable_id_and_finite_metrics() -> None:
    first = build_fixture_report()
    second = build_fixture_report()
    assert first["report_id"] == second["report_id"]
    assert len(first["report_id"]) == 16
    assert first["report_id"] == compute_report_id(first)
    assert all(0.0 <= value <= 1.0 for value in first["metrics"].values())  # type: ignore[union-attr]


def test_fixture_report_id_binds_detailed_diagnostics() -> None:
    report = build_fixture_report()
    original_id = report["report_id"]
    report["details"] = {"tampered": True}
    assert compute_report_id(report) != original_id


def test_fixture_report_cannot_pass_publication_gate() -> None:
    payload = build_fixture_report()
    report = EvaluationReport.model_validate(
        {key: payload[key] for key in EvaluationReport.model_fields}
    )
    with pytest.raises(PermissionError, match="not authorized"):
        assert_publication_authorized(report)


def test_authorized_frozen_report_can_pass_gate() -> None:
    report = EvaluationReport(
        schema_version="roadsense.evaluation-report/v1",
        protocol_id="authorized/v1",
        evidence_level=EvidenceLevel.FROZEN_EVALUATION,
        dataset_manifest_sha256="a" * 64,
        evaluation_authorized=True,
        frozen=True,
        metrics={"metric": 0.5},
        claim_boundary="authorized test fixture",
    )
    assert_publication_authorized(report)
