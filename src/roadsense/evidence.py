"""Evidence reports and publication gates."""

from __future__ import annotations

from roadsense.contracts import EvaluationReport, EvidenceLevel
from roadsense.fixture import build_fixture_bundle, build_fixture_metrics
from roadsense.json_io import canonical_sha256


def compute_report_id(payload: dict[str, object]) -> str:
    """Return the short canonical identity for a complete report payload.

    ``report_id`` is excluded so the identifier can be recomputed during API
    validation. Every other public field, including the detailed metric
    diagnostics, is bound by the digest.
    """

    material = {key: value for key, value in payload.items() if key != "report_id"}
    return canonical_sha256(material)[:16]


def build_fixture_report() -> dict[str, object]:
    bundle = build_fixture_bundle()
    metrics = build_fixture_metrics(bundle)
    flattened = {
        "detection_ap50": float(metrics["detection"]["ap"]),  # type: ignore[index]
        "detection_precision": float(metrics["detection"]["precision"]),  # type: ignore[index]
        "detection_recall": float(metrics["detection"]["recall"]),  # type: ignore[index]
        "segmentation_mean_iou": float(metrics["segmentation"]["mean_iou"]),  # type: ignore[index]
        "segmentation_pixel_accuracy": float(
            metrics["segmentation"]["pixel_accuracy"]  # type: ignore[index]
        ),
        "tracking_mota": float(metrics["tracking"]["mota"]),  # type: ignore[index]
        "tracking_identity_f1": float(metrics["tracking"]["identity_f1"]),  # type: ignore[index]
    }
    report = EvaluationReport(
        schema_version="roadsense.evaluation-report/v1",
        protocol_id="roadsense.fixture-city-loop/v4",
        evidence_level=EvidenceLevel.FIXTURE,
        evaluation_authorized=False,
        frozen=False,
        metrics=flattened,
        claim_boundary=(
            "Deterministic fixture metrics validate implementation contracts only. "
            "No public-dataset quality, runtime, robustness, or production claim is authorized."
        ),
    ).model_dump(mode="json")
    report["details"] = metrics
    report["report_id"] = compute_report_id(report)
    return report


def assert_publication_authorized(report: EvaluationReport) -> None:
    if (
        report.evidence_level is not EvidenceLevel.FROZEN_EVALUATION
        or not report.evaluation_authorized
        or not report.frozen
        or report.dataset_manifest_sha256 is None
    ):
        raise PermissionError("report is not authorized frozen evaluation evidence")
