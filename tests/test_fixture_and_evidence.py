from __future__ import annotations

from itertools import combinations, pairwise

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


def test_all_scene_cars_keep_their_full_box_inside_the_road() -> None:
    payload = build_demo_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)

    for frame in frames:
        road = next(segment for segment in frame["segments"] if segment["label"] == "road")
        polygon = road["polygon"]
        for item in frame["actors"]:
            if item["label"] != "car":
                continue
            x, y, width, height = item["bbox"]
            for edge_y in (y, y + height):
                intersections: list[float] = []
                closed_polygon = [*polygon, polygon[0]]
                for start, end in pairwise(closed_polygon):
                    x1, y1 = start
                    x2, y2 = end
                    if y1 == y2 or not min(y1, y2) <= edge_y <= max(y1, y2):
                        continue
                    progress = (edge_y - y1) / (y2 - y1)
                    intersections.append(x1 + (x2 - x1) * progress)
                assert len(intersections) >= 2
                left, right = min(intersections), max(intersections)
                assert left <= x
                assert x + width <= right


def test_scene_cars_do_not_overlap_pedestrians_or_cyclists() -> None:
    payload = build_demo_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)

    for frame in frames:
        for first, second in combinations(frame["actors"], 2):
            labels = {first["label"], second["label"]}
            if "car" not in labels or labels.isdisjoint({"pedestrian", "cyclist"}):
                continue
            first_x, first_y, first_width, first_height = first["bbox"]
            second_x, second_y, second_width, second_height = second["bbox"]
            overlap_width = max(
                0,
                min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
            )
            overlap_height = max(
                0,
                min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
            )
            assert overlap_width * overlap_height == 0


def test_scene_actors_use_continuous_truth_tracks_while_predictions_keep_errors() -> None:
    payload = build_demo_payload()
    frames = payload["frames"]
    assert isinstance(frames, list)

    actor_tracks: dict[int, list[tuple[int, float, float]]] = {}
    prediction_presence: dict[int, set[int]] = {}
    for frame in frames:
        frame_index = frame["frame_index"]
        for item in frame["actors"]:
            x, y, width, height = item["bbox"]
            actor_tracks.setdefault(item["track_id"], []).append(
                (frame_index, x + width / 2, y + height)
            )
        for item in frame["objects"]:
            prediction_presence.setdefault(item["track_id"], set()).add(frame_index)

    for positions in actor_tracks.values():
        assert all(next_frame == frame + 1 for (frame, *_), (next_frame, *_) in pairwise(positions))

    for track_id in VEHICLE_TRACK_IDS:
        positions = actor_tracks[track_id]
        assert all(next_bottom > bottom for (*_, bottom), (*_, next_bottom) in pairwise(positions))
        assert all(
            abs(next_center_x - center_x) < next_bottom - bottom
            for (_frame, center_x, bottom), (_next_frame, next_center_x, next_bottom) in pairwise(
                positions
            )
        )

    # The scene remains physically continuous while the prediction layer
    # deliberately contains misses and an identity switch for metric plumbing.
    assert 15 in {frame for frame, *_ in actor_tracks[102]}
    assert 15 not in prediction_presence[2]
    assert {10, 11, 12}.issubset({frame for frame, *_ in actor_tracks[201]})
    assert {10, 11, 12}.isdisjoint(prediction_presence[3])


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
