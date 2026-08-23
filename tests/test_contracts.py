from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from roadsense.contracts import (
    BoxXYXY,
    DatasetManifest,
    Detection,
    EvaluationReport,
    EvidenceLevel,
    FrameRecord,
    ImageSize,
)


def test_box_properties() -> None:
    box = BoxXYXY(x_min=2, y_min=3, x_max=12, y_max=8)
    assert box.width == 10
    assert box.height == 5
    assert box.area == 50


@pytest.mark.parametrize(
    "payload",
    [
        {"x_min": 1, "y_min": 1, "x_max": 1, "y_max": 2},
        {"x_min": 2, "y_min": 1, "x_max": 1, "y_max": 2},
        {"x_min": 0, "y_min": 0, "x_max": math.inf, "y_max": 2},
    ],
)
def test_box_rejects_invalid_geometry(payload: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        BoxXYXY(**payload)


def test_frame_rejects_out_of_bounds_detection() -> None:
    with pytest.raises(ValidationError, match="outside the image"):
        FrameRecord(
            frame_index=0,
            timestamp_ms=0,
            image_size=ImageSize(width=10, height=10),
            detections=(
                Detection(
                    category_id=1,
                    score=0.5,
                    bbox=BoxXYXY(x_min=1, y_min=1, x_max=11, y_max=5),
                ),
            ),
        )


def test_detection_rejects_non_finite_score() -> None:
    with pytest.raises(ValidationError):
        Detection(
            category_id=1,
            score=math.nan,
            bbox=BoxXYXY(x_min=1, y_min=1, x_max=2, y_max=2),
        )


@pytest.mark.parametrize("value", [True, "10", 10.0])
def test_image_size_rejects_integer_coercion(value: object) -> None:
    with pytest.raises(ValidationError, match="integer"):
        ImageSize(width=value, height=10)


@pytest.mark.parametrize("value", [True, "0.5"])
def test_box_rejects_numeric_coercion(value: object) -> None:
    with pytest.raises(ValidationError, match="number"):
        BoxXYXY(x_min=value, y_min=0, x_max=1, y_max=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category_id", True),
        ("category_id", "1"),
        ("score", True),
        ("score", "0.5"),
        ("track_id", True),
        ("track_id", "7"),
    ],
)
def test_detection_rejects_numeric_coercion(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "category_id": 1,
        "score": 0.5,
        "bbox": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    }
    payload[field] = value
    with pytest.raises(ValidationError, match="(integer|number)"):
        Detection.model_validate(payload)


@pytest.mark.parametrize("field", ["frame_index", "timestamp_ms"])
def test_frame_rejects_temporal_numeric_coercion(field: str) -> None:
    payload: dict[str, object] = {
        "frame_index": 0,
        "timestamp_ms": 0,
        "image_size": {"width": 10, "height": 10},
        "detections": [],
    }
    payload[field] = "0"
    with pytest.raises(ValidationError, match="integer"):
        FrameRecord.model_validate(payload)


def test_numeric_contracts_accept_numpy_scalars() -> None:
    frame = FrameRecord(
        frame_index=np.int64(0),
        timestamp_ms=np.uint32(0),
        image_size=ImageSize(width=np.int32(10), height=np.int32(10)),
        detections=(
            Detection(
                category_id=np.int64(1),
                score=np.float32(0.5),
                track_id=np.uint32(3),
                bbox=BoxXYXY(
                    x_min=np.float32(0),
                    y_min=np.float64(0),
                    x_max=np.float32(1),
                    y_max=np.float64(1),
                ),
            ),
        ),
    )
    assert frame.frame_index == 0
    assert frame.detections[0].track_id == 3


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "roadsense.dataset-manifest/v1",
        "dataset_name": "fixture",
        "source_url": "repository://fixture",
        "license_id": "MIT",
        "tasks": ["detection", "tracking"],
        "splits": {"dev": "fixed"},
        "content_sha256": "1" * 64,
        "evaluation_authorized": False,
        "frozen": False,
    }


def test_manifest_accepts_explicit_unscored_fixture() -> None:
    manifest = DatasetManifest.model_validate(_manifest_payload())
    assert [task.value for task in manifest.tasks] == ["detection", "tracking"]


def test_manifest_rejects_duplicate_tasks() -> None:
    payload = _manifest_payload()
    payload["tasks"] = ["detection", "detection"]
    with pytest.raises(ValidationError, match="unique"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_frozen_without_authorization() -> None:
    payload = _manifest_payload()
    payload["frozen"] = True
    with pytest.raises(ValidationError, match="authorized"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_placeholder_content_hash() -> None:
    payload = _manifest_payload()
    payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="placeholder"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_coerced_evidence_flags() -> None:
    payload = _manifest_payload()
    payload["evaluation_authorized"] = "false"
    with pytest.raises(ValidationError, match="boolean"):
        DatasetManifest.model_validate(payload)


def test_frozen_report_requires_manifest_hash() -> None:
    with pytest.raises(ValidationError, match="manifest"):
        EvaluationReport(
            schema_version="roadsense.evaluation-report/v1",
            protocol_id="test/v1",
            evidence_level=EvidenceLevel.FROZEN_EVALUATION,
            evaluation_authorized=True,
            frozen=True,
            metrics={"ap": 0.5},
            claim_boundary="test only",
        )


def test_fixture_report_cannot_be_marked_authorized_or_frozen() -> None:
    with pytest.raises(ValidationError, match="fixture evidence"):
        EvaluationReport(
            schema_version="roadsense.evaluation-report/v1",
            protocol_id="test/v1",
            evidence_level=EvidenceLevel.FIXTURE,
            evaluation_authorized=True,
            frozen=False,
            metrics={"ap": 0.5},
            claim_boundary="test only",
        )


def test_development_report_cannot_be_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen_evaluation"):
        EvaluationReport(
            schema_version="roadsense.evaluation-report/v1",
            protocol_id="test/v1",
            evidence_level=EvidenceLevel.DEVELOPMENT,
            evaluation_authorized=True,
            frozen=True,
            dataset_manifest_sha256="a" * 64,
            metrics={"ap": 0.5},
            claim_boundary="test only",
        )


def test_evaluation_report_requires_named_metrics() -> None:
    with pytest.raises(ValidationError, match="named metrics"):
        EvaluationReport(
            schema_version="roadsense.evaluation-report/v1",
            protocol_id="test/v1",
            evidence_level=EvidenceLevel.FIXTURE,
            evaluation_authorized=False,
            frozen=False,
            metrics={},
            claim_boundary="test only",
        )


def test_frozen_contract_containers_reject_in_place_mutation() -> None:
    manifest = DatasetManifest.model_validate(_manifest_payload())
    with pytest.raises(TypeError, match="immutable"):
        manifest.splits["dev"] = "tampered"

    report = EvaluationReport(
        schema_version="roadsense.evaluation-report/v1",
        protocol_id="test/v1",
        evidence_level=EvidenceLevel.FIXTURE,
        evaluation_authorized=False,
        frozen=False,
        metrics={"ap": 0.5},
        claim_boundary="test only",
    )
    with pytest.raises(TypeError, match="immutable"):
        report.metrics["ap"] = 0.0
