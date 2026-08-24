"""Pure-function coverage for the optional COCO8 ONNX runner.

These tests intentionally do not construct an ONNX Runtime session or access
the downloaded real-data directory.  They protect the adapter boundary where
YOLO text labels and tensor outputs are converted into RoadSense detections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from scripts.run_coco8_onnx_eval import (
    _decode_predictions,
    _letterbox,
    _load_truth,
    _tree_inventory,
    _validate_onnx_session,
)


def test_load_truth_converts_normalized_yolo_box(tmp_path: Path) -> None:
    labels = tmp_path / "sample.txt"
    labels.write_text("2 0.5 0.5 0.5 0.25\n", encoding="utf-8")

    detections = _load_truth(labels, width=200, height=100)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.category_id == 2
    assert detection.score == 1.0
    assert detection.bbox.model_dump() == {
        "x_min": 50.0,
        "y_min": 37.5,
        "x_max": 150.0,
        "y_max": 62.5,
    }


@pytest.mark.parametrize("size, expected_pad", [((481, 640), (79, 0)), ((640, 481), (0, 79))])
def test_letterbox_uses_actual_integer_padding(
    tmp_path: Path, size: tuple[int, int], expected_pad: tuple[int, int]
) -> None:
    image_path = tmp_path / "solid.png"
    Image.new("RGB", size, (255, 0, 0)).save(image_path)

    tensor, scale, pad_x, pad_y, width, height = _letterbox(image_path)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert (pad_x, pad_y) == expected_pad
    assert (width, height) == size
    assert scale == pytest.approx(1.0)
    # The first source pixel is exactly at the integer paste offset; the
    # preceding pixel remains the 114-valued canvas border.
    x, y = expected_pad
    assert tensor[0, :, y, x].tolist() == pytest.approx([1.0, 0.0, 0.0])
    if x:
        assert tensor[0, :, y, x - 1].tolist() == pytest.approx([114 / 255, 114 / 255, 114 / 255])
    else:
        assert tensor[0, :, y - 1, x].tolist() == pytest.approx([114 / 255, 114 / 255, 114 / 255])


def _yolo_output(*, transposed: bool) -> np.ndarray:
    """Build three boxes: one cross-class pair and one same-class duplicate."""

    values = np.zeros((3, 84), dtype=np.float32)
    # class 2: duplicate boxes overlap at IoU > .5, so only the .80 box stays
    values[0, :4] = (100, 100, 40, 40)
    values[0, 4 + 2] = 0.80
    values[1, :4] = (101, 101, 40, 40)
    values[1, 4 + 2] = 0.70
    # class 1: same geometry but a distinct category must not be suppressed
    values[2, :4] = (100, 100, 40, 40)
    values[2, 4 + 1] = 0.95
    # ``transposed=False`` is [batch, 84, anchors], while ``True`` is the
    # alternate [batch, anchors, 84] representation accepted by the decoder.
    return values[None] if transposed else values[None].transpose(0, 2, 1)


@pytest.mark.parametrize("transposed", [False, True])
def test_decode_predictions_accepts_yolo_layout_and_class_aware_nms(transposed: bool) -> None:
    # The runner accepts the canonical [1, 84, anchors] layout and transposes
    # it internally.  Keep a separate parameter as a regression guard for the
    # orientation branch; the second case supplies [1, anchors, 84].
    values = _yolo_output(transposed=transposed)

    detections = _decode_predictions(
        values,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        width=640,
        height=640,
        score_threshold=0.5,
        nms_iou=0.5,
        max_detections=10,
    )

    assert [item.category_id for item in detections] == [1, 2]
    assert [item.score for item in detections] == pytest.approx([0.95, 0.80])


def test_decode_predictions_rejects_nonfinite_model_output() -> None:
    values = _yolo_output(transposed=False)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _decode_predictions(
            values,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            width=640,
            height=640,
            score_threshold=0.5,
            nms_iou=0.5,
            max_detections=10,
        )


def test_decode_predictions_rejects_invalid_transform_and_scores() -> None:
    values = _yolo_output(transposed=False)
    with pytest.raises(ValueError, match="scale"):
        _decode_predictions(
            values,
            scale=0.0,
            pad_x=0,
            pad_y=0,
            width=640,
            height=640,
            score_threshold=0.5,
            nms_iou=0.5,
            max_detections=10,
        )

    values[0, 4 + 2, 0] = 1.1
    with pytest.raises(ValueError, match="class scores"):
        _decode_predictions(
            values,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            width=640,
            height=640,
            score_threshold=0.5,
            nms_iou=0.5,
            max_detections=10,
        )


def test_decode_predictions_rejects_invalid_padding_types() -> None:
    values = _yolo_output(transposed=False)
    with pytest.raises((TypeError, ValueError), match="pad_x"):
        _decode_predictions(
            values,
            scale=1.0,
            pad_x=-1,
            pad_y=0,
            width=640,
            height=640,
            score_threshold=0.5,
            nms_iou=0.5,
            max_detections=10,
        )


def test_tree_inventory_rejects_symlink_entries(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        _tree_inventory(tmp_path)


def test_tree_inventory_can_exclude_generated_output_subtree(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "report.json").write_text("generated", encoding="utf-8")

    inventory, _ = _tree_inventory(tmp_path, exclude=output_dir)

    assert [item["path"] for item in inventory] == ["source.txt"]


class _FakeMeta:
    def __init__(self, name: str, tensor_type: str, shape: list[object]) -> None:
        self.name = name
        self.type = tensor_type
        self.shape = shape


class _FakeSession:
    def __init__(self, inputs: list[_FakeMeta], outputs: list[_FakeMeta]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> list[_FakeMeta]:
        return self._inputs

    def get_outputs(self) -> list[_FakeMeta]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def test_validate_onnx_session_binds_graph_contract() -> None:
    session = _FakeSession(
        [_FakeMeta("images", "tensor(float)", ["batch", 3, "height", "width"])],
        [_FakeMeta("output0", "tensor(float)", ["batch", 84, "anchors"])],
    )
    inputs, outputs = _validate_onnx_session(session)
    assert inputs.name == "images"
    assert outputs.name == "output0"

    bad = _FakeSession(
        [_FakeMeta("input", "tensor(float)", ["batch", 3, "height", "width"])],
        [_FakeMeta("output0", "tensor(float)", ["batch", 84, "anchors"])],
    )
    with pytest.raises(ValueError, match="tensor names"):
        _validate_onnx_session(bad)
