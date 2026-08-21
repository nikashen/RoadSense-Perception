from __future__ import annotations

import math

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
