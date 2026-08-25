"""Unit coverage for offline, fail-closed BDD100K Detection preparation."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest

import scripts.prepare_bdd100k_detection as preparation_module
from roadsense.contracts import DatasetManifest
from roadsense.json_io import canonical_sha256, load_strict_json
from scripts.prepare_bdd100k_detection import (
    BDD100K_DETECTION_CATEGORIES,
    DEFAULT_EXPECTED_IMAGE_COUNT,
    IMAGE_MANIFEST_SCHEMA,
    BDDPreparationError,
    _build_parser,
    _validate_det_val_frames,
    main,
    prepare_bdd100k_detection,
)


def _write_zip(path: Path, entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in entries:
            archive.writestr(member, payload)
    return path


def _frame(name: str, categories: list[str]) -> dict[str, object]:
    return {
        "name": name,
        "labels": [
            {
                "category": category,
                "box2d": {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 12.0},
            }
            for category in categories
        ],
    }


def _valid_frames(names: list[str]) -> list[dict[str, object]]:
    """Place all official categories across an arbitrary small synthetic split."""

    result = [_frame(name, []) for name in names]
    for index, category in enumerate(BDD100K_DETECTION_CATEGORIES):
        labels = result[index % len(result)]["labels"]
        assert isinstance(labels, list)
        labels.append(
            {
                "category": category,
                "box2d": {"x1": 1, "y1": 2, "x2": 11, "y2": 12},
            }
        )
    return result


def _sources(tmp_path: Path, *, names: list[str] | None = None) -> tuple[Path, Path, list[str]]:
    image_names = names or ["z-last.jpg", "a-first.jpg"]
    images_zip = _write_zip(
        tmp_path / "bdd100k_images_100k_val.zip",
        [
            (f"bdd100k/images/100k/val/{name}", f"synthetic:{name}".encode("ascii"))
            for name in image_names
        ],
    )
    labels = json.dumps(_valid_frames(image_names), sort_keys=True).encode("utf-8")
    labels_zip = _write_zip(
        tmp_path / "det_20_labels_val.zip",
        [("det_20/det_val.json", labels)],
    )
    return images_zip, labels_zip, image_names


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _prepare(tmp_path: Path, **overrides: Any) -> tuple[Path, dict[str, object]]:
    images_zip, labels_zip, names = _sources(tmp_path)
    data_root = tmp_path / "data" / "raw" / "bdd100k_det20_val"
    kwargs: dict[str, Any] = {
        "images_zip": images_zip,
        "labels_input": labels_zip,
        "data_root": data_root,
        "expected_image_count": len(names),
        "images_official_md5": _md5(images_zip),
        "labels_official_md5": _md5(labels_zip),
        "accept_bdd100k_research_license": True,
    }
    kwargs.update(overrides)
    return data_root, prepare_bdd100k_detection(**kwargs)


def test_prepare_writes_relative_auditable_image_manifest_and_receipts(tmp_path: Path) -> None:
    data_root, result = _prepare(tmp_path)

    assert result["image_manifest"] == "image-manifest.json"
    image_manifest = load_strict_json(data_root / "image-manifest.json")
    assert set(image_manifest) == {
        "schema_version",
        "dataset_id",
        "task",
        "split",
        "image_count",
        "images_tree_sha256",
        "images",
        "source_archives",
    }
    assert image_manifest["schema_version"] == IMAGE_MANIFEST_SCHEMA
    assert image_manifest["dataset_id"] == "BDD100K"
    assert image_manifest["task"] == "detection"
    assert image_manifest["split"] == "val"
    assert image_manifest["image_count"] == 2
    assert [item["name"] for item in image_manifest["images"]] == [
        "a-first.jpg",
        "z-last.jpg",
    ]
    assert image_manifest["images_tree_sha256"] == canonical_sha256(image_manifest["images"])
    assert [item["official_package_md5"] for item in image_manifest["source_archives"]] == [
        _md5(tmp_path / "bdd100k_images_100k_val.zip"),
        _md5(tmp_path / "det_20_labels_val.zip"),
    ]

    inventory = load_strict_json(data_root / "split-inventory.json")
    assert inventory["image_manifest"] == "image-manifest.json"
    assert inventory["labels"]["frame_count"] == 2
    assert inventory["labels"]["categories"] == list(BDD100K_DETECTION_CATEGORIES)
    assert (data_root / "images" / "val" / "a-first.jpg").read_bytes() == b"synthetic:a-first.jpg"
    assert (data_root / "labels" / "det_val.json").is_file()

    manifest = DatasetManifest.model_validate(load_strict_json(data_root / "dataset-manifest.json"))
    assert manifest.evaluation_authorized is True
    assert manifest.frozen is False
    source_receipt = load_strict_json(data_root / "source-receipt.json")
    assert source_receipt["local_only"] is True
    assert source_receipt["prepared_layout"]["images"] == "images/val"

    # Output JSON is portable evidence: no temporary/local absolute paths leak.
    absolute_root = str(tmp_path.resolve())
    for filename in (
        "image-manifest.json",
        "split-inventory.json",
        "dataset-manifest.json",
        "source-receipt.json",
    ):
        assert absolute_root not in (data_root / filename).read_text(encoding="utf-8")


def test_prepare_accepts_an_already_extracted_local_det_val_json(tmp_path: Path) -> None:
    images_zip, _labels_zip, names = _sources(tmp_path)
    labels_path = tmp_path / "det_val.json"
    labels_path.write_text(json.dumps(_valid_frames(names)), encoding="utf-8")
    data_root = tmp_path / "prepared"

    prepare_bdd100k_detection(
        images_zip=images_zip,
        labels_input=labels_path,
        data_root=data_root,
        expected_image_count=len(names),
        accept_bdd100k_research_license=True,
    )

    receipt = load_strict_json(data_root / "source-receipt.json")
    assert receipt["source_archives"][1]["format"] == "json"
    assert (data_root / "labels" / "det_val.json").read_bytes() == labels_path.read_bytes()


def test_prepare_requires_explicit_license_acknowledgement(tmp_path: Path) -> None:
    images_zip, labels_zip, names = _sources(tmp_path)
    data_root = tmp_path / "prepared"

    with pytest.raises(BDDPreparationError, match="accept-bdd100k-research-license"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=data_root,
            expected_image_count=len(names),
        )
    assert not data_root.exists()


def test_cli_defaults_to_official_count_and_refuses_missing_license(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["--images-zip", "images.zip", "--labels", "labels.zip"])
    assert parsed.expected_image_count == DEFAULT_EXPECTED_IMAGE_COUNT

    images_zip, labels_zip, names = _sources(tmp_path)
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "--images-zip",
                str(images_zip),
                "--labels-zip",
                str(labels_zip),
                "--data-root",
                str(output),
                "--expected-image-count",
                str(len(names)),
            ]
        )
        == 2
    )
    assert "accept-bdd100k-research-license" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("member_name", "expected_message"),
    [
        ("../outside.jpg", "traversal"),
        ("bdd100k/images/100k/val/./duplicate.jpg", "traversal"),
    ],
)
def test_prepare_rejects_unsafe_zip_member_paths(
    tmp_path: Path, member_name: str, expected_message: str
) -> None:
    images_zip, labels_zip, names = _sources(tmp_path)
    with zipfile.ZipFile(images_zip, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"unsafe")

    with pytest.raises(BDDPreparationError, match=expected_message):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "prepared",
            expected_image_count=len(names),
            accept_bdd100k_research_license=True,
        )


def test_prepare_rejects_zip_symlink_and_leaves_no_output(tmp_path: Path) -> None:
    images_zip, labels_zip, names = _sources(tmp_path)
    link = zipfile.ZipInfo("bdd100k/images/100k/val/link.jpg")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(images_zip, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(link, b"target")
    output = tmp_path / "prepared"

    with pytest.raises(BDDPreparationError, match="symbolic-link"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=output,
            expected_image_count=len(names),
            accept_bdd100k_research_license=True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.prepare-*"))


def test_prepare_rejects_duplicate_zip_paths_case_insensitively(tmp_path: Path) -> None:
    images_zip, labels_zip, names = _sources(tmp_path)
    with zipfile.ZipFile(images_zip, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BDD100K/IMAGES/100K/VAL/A-FIRST.JPG", b"duplicate")

    with pytest.raises(BDDPreparationError, match="duplicate"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "prepared",
            expected_image_count=len(names),
            accept_bdd100k_research_license=True,
        )


def test_det_val_validation_requires_exact_images_and_allows_absent_categories() -> None:
    with pytest.raises(BDDPreparationError, match="exactly match"):
        _validate_det_val_frames(
            _valid_frames(["other.jpg", "b.jpg"]),
            image_names=("a.jpg", "b.jpg"),
            expected_image_count=2,
        )

    incomplete = [_frame("a.jpg", ["car"]), _frame("b.jpg", ["truck"])]
    summary = _validate_det_val_frames(
        incomplete,
        image_names=("a.jpg", "b.jpg"),
        expected_image_count=2,
    )
    assert summary["categories"] == list(BDD100K_DETECTION_CATEGORIES)

    unknown = [_frame("a.jpg", ["car"]), _frame("b.jpg", ["alien"])]
    with pytest.raises(BDDPreparationError, match="non-BDD100K category"):
        _validate_det_val_frames(
            unknown,
            image_names=("a.jpg", "b.jpg"),
            expected_image_count=2,
        )


def test_prepare_checks_optional_official_package_md5_before_extracting(tmp_path: Path) -> None:
    images_zip, labels_zip, names = _sources(tmp_path)
    with pytest.raises(BDDPreparationError, match="MD5"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "prepared",
            expected_image_count=len(names),
            images_official_md5="0" * 32,
            accept_bdd100k_research_license=True,
        )


def test_prepare_formal_lane_uses_both_published_package_md5s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images_zip, labels_zip, _names = _sources(tmp_path)

    with pytest.raises(BDDPreparationError, match="images package MD5"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "missing-images-md5",
            expected_image_count=DEFAULT_EXPECTED_IMAGE_COUNT,
            accept_bdd100k_research_license=True,
        )

    with pytest.raises(BDDPreparationError, match="--labels-md5 must equal"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "wrong-labels-md5",
            expected_image_count=DEFAULT_EXPECTED_IMAGE_COUNT,
            images_official_md5=preparation_module.BDD100K_OFFICIAL_IMAGES_MD5,
            labels_official_md5="0" * 32,
            accept_bdd100k_research_license=True,
        )

    observed_md5 = {
        images_zip: preparation_module.BDD100K_OFFICIAL_IMAGES_MD5,
        labels_zip: "0" * 32,
    }

    def fake_hash_file(path: Path, *, include_md5: bool) -> tuple[str, int, str | None]:
        assert include_md5 is True
        return "1" * 64, path.stat().st_size, observed_md5[path]

    monkeypatch.setattr(preparation_module, "_hash_file", fake_hash_file)
    with pytest.raises(BDDPreparationError, match="labels package MD5"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "missing-labels-md5",
            expected_image_count=DEFAULT_EXPECTED_IMAGE_COUNT,
            accept_bdd100k_research_license=True,
        )

    with pytest.raises(BDDPreparationError, match="--images-md5 must equal"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_zip,
            data_root=tmp_path / "wrong-images-md5",
            expected_image_count=DEFAULT_EXPECTED_IMAGE_COUNT,
            images_official_md5="0" * 32,
            labels_official_md5=preparation_module.BDD100K_OFFICIAL_LABELS_MD5,
            accept_bdd100k_research_license=True,
        )


def test_prepare_formal_lane_rejects_extracted_or_community_labels(tmp_path: Path) -> None:
    images_zip, _labels_zip, _names = _sources(tmp_path)
    labels_json = tmp_path / "det_val.json"
    labels_json.write_text("[]", encoding="utf-8")

    with pytest.raises(BDDPreparationError, match="official labels ZIP"):
        prepare_bdd100k_detection(
            images_zip=images_zip,
            labels_input=labels_json,
            data_root=tmp_path / "community-labels",
            expected_image_count=DEFAULT_EXPECTED_IMAGE_COUNT,
            images_official_md5=preparation_module.BDD100K_OFFICIAL_IMAGES_MD5,
            labels_official_md5=preparation_module.BDD100K_OFFICIAL_LABELS_MD5,
            accept_bdd100k_research_license=True,
        )
