"""Prepare an auditable local BDD100K Detection 2020 validation bundle.

The BDD100K images and annotations are intentionally operator-provided inputs.
This module never downloads data.  It requires an explicit acknowledgement of
the BDD100K research licence, verifies the supplied archives before extracting
them, and writes only relative-path evidence records into the requested local
data root.  The resulting files are designed to be consumed by the local
benchmark runner; neither the images nor the annotations are public assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from roadsense.contracts import DatasetManifest
from roadsense.json_io import canonical_sha256, load_strict_json, write_json_atomic

DEFAULT_EXPECTED_IMAGE_COUNT = 10_000
DATASET_ID = "BDD100K"
TASK = "detection"
SPLIT = "val"
IMAGE_MANIFEST_SCHEMA = "roadsense.bdd100k-detection-images/v1"
SPLIT_INVENTORY_SCHEMA = "roadsense.bdd100k-detection-split-inventory/v1"
SOURCE_RECEIPT_SCHEMA = "roadsense.bdd100k-detection-source-receipt/v1"

# The official package uses this layout for its validation members.
# ``images/100k/val`` is accepted as the equivalent layout emitted by mirrors
# which omit the top-level ``bdd100k`` directory.
IMAGE_LAYOUT_PREFIXES = (
    "bdd100k/images/100k/val",
    "images/100k/val",
    "100k/val",
)
LABEL_MEMBER_PATHS = (
    "det_20/det_val.json",
    "labels/det_20/det_val.json",
    "bdd100k/labels/det_20/det_val.json",
)
BDD100K_DETECTION_CATEGORIES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)
BDD100K_SOURCE_URL = "https://bdd-data.berkeley.edu/"
# The Berkeley download page currently exposes these packages through its
# official HTTP mirror.  The images package contains train/val/test; the
# preparation step selects and hashes only ``images/100k/val``.  Keep the
# source URLs in the local receipt so a future HTTPS portal can be substituted
# only with a deliberate new receipt.
BDD100K_IMAGES_VAL_URL = "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
BDD100K_DET20_VAL_LABELS_URL = "http://128.32.162.150/bdd100k/bdd100k_det_20_labels.zip"

# The official val image package is large.  These are deliberately generous
# limits that protect against malformed archives without excluding it.
# The labels package may contain both train and validation JSON members; the
# unused training member can be larger than 512 MiB after decompression.
MAX_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
# The official 100K image archive includes train/val/test members.  Its
# uncompressed size is substantially larger than the published 5.3 GB ZIP;
# this bound leaves room for that archive while still rejecting pathological
# ZIP bombs before extraction.
MAX_TOTAL_UNCOMPRESSED_BYTES = 128 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 200_000
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class BDDPreparationError(ValueError):
    """Raised when an operator-provided BDD100K input fails a closed check."""


def _reject_windows_path_hazards(parts: Iterable[str], *, role: str) -> None:
    """Reject names that can alias or escape when materialized on Windows."""

    for part in parts:
        if ":" in part or part.endswith((".", " ")):
            raise BDDPreparationError(f"{role} contains a Windows-unsafe member name")
        if part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise BDDPreparationError(f"{role} contains a reserved Windows member name")


def _hash_file(path: Path, *, include_md5: bool = False) -> tuple[str, int, str | None]:
    """Return SHA-256, byte count, and optionally MD5 in one streaming pass."""

    sha256 = hashlib.sha256()
    md5 = hashlib.md5() if include_md5 else None
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                sha256.update(chunk)
                if md5 is not None:
                    md5.update(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise BDDPreparationError(f"cannot read local input: {path.name}") from exc
    return sha256.hexdigest(), byte_count, None if md5 is None else md5.hexdigest()


def _normalise_expected_md5(value: str | None, *, option: str) -> str | None:
    """Validate an optional official package MD5 without treating it as trust."""

    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 32:
        raise BDDPreparationError(f"{option} must be a 32-character hexadecimal MD5")
    normalised = value.lower()
    if any(character not in "0123456789abcdef" for character in normalised):
        raise BDDPreparationError(f"{option} must be a 32-character hexadecimal MD5")
    return normalised


def _verify_official_md5(observed: str | None, expected: str | None, *, role: str) -> None:
    if expected is not None and observed != expected:
        raise BDDPreparationError(f"{role} MD5 does not match the supplied official package MD5")


def _zip_member_path(info: zipfile.ZipInfo) -> tuple[str, bool]:
    """Return a safe, portable archive member path and whether it is a directory.

    ZIP filenames use POSIX separators, but rejecting backslashes and Windows
    drive-qualified spellings is necessary when extraction runs on Windows.
    """

    raw_name = info.filename
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise BDDPreparationError("archive contains an unsafe member name")
    if raw_name.startswith("/") or PureWindowsPath(raw_name).drive:
        raise BDDPreparationError("archive contains an absolute member path")
    raw_parts = raw_name.rstrip("/").split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise BDDPreparationError("archive contains a traversal member path")
    pure_path = PurePosixPath(raw_name)
    if pure_path.is_absolute() or not pure_path.parts:
        raise BDDPreparationError("archive contains an unsafe member path")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise BDDPreparationError("archive contains a traversal member path")
    if any(
        any(ord(character) < 32 or ord(character) == 127 for character in part)
        for part in pure_path.parts
    ):
        raise BDDPreparationError("archive member path contains a control character")
    _reject_windows_path_hazards(pure_path.parts, role="archive member path")
    normalised = "/".join(pure_path.parts)

    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise BDDPreparationError("archive contains a symbolic-link member")
    is_directory = raw_name.endswith("/") or file_type == stat.S_IFDIR
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise BDDPreparationError("archive contains a non-regular member")
    if is_directory and not raw_name.endswith("/"):
        raise BDDPreparationError("archive directory member must end with '/'")
    return normalised, is_directory


def _validated_zip_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str, bool]]:
    """Validate archive paths, links, collisions, and resource declarations."""

    infos = archive.infolist()
    if not infos:
        raise BDDPreparationError("archive is empty")
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise BDDPreparationError("archive contains too many members")

    members: list[tuple[zipfile.ZipInfo, str, bool]] = []
    seen: dict[str, bool] = {}
    total_uncompressed = 0
    for info in infos:
        if info.flag_bits & 0x1:
            raise BDDPreparationError("encrypted archive members are not accepted")
        path, is_directory = _zip_member_path(info)
        key = path.casefold()
        if key in seen:
            raise BDDPreparationError("archive contains duplicate member paths")
        seen[key] = is_directory
        if not is_directory:
            if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                raise BDDPreparationError("archive member exceeds the safe size limit")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise BDDPreparationError("archive exceeds the safe uncompressed size limit")
        members.append((info, path, is_directory))

    regular_paths = {path.casefold() for _, path, is_directory in members if not is_directory}
    for _, path, _ in members:
        ancestor = PurePosixPath(path)
        for parent in ancestor.parents:
            if str(parent) != "." and str(parent).casefold() in regular_paths:
                raise BDDPreparationError("archive has a file/directory path collision")
    return members


def _safe_extract_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path
) -> tuple[str, int]:
    """Extract one prevalidated regular member and hash its actual bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with archive.open(info, "r") as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
                byte_count += len(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BDDPreparationError("archive extraction failed") from exc
    if byte_count != info.file_size:
        raise BDDPreparationError("archive member size changed while extracting")
    return digest.hexdigest(), byte_count


def _safe_copy_file(source: Path, destination: Path) -> tuple[str, int]:
    """Copy a local JSON label file without retaining its absolute source path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                digest.update(chunk)
                output_handle.write(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise BDDPreparationError("cannot copy local det_val JSON") from exc
    return digest.hexdigest(), byte_count


def _is_safe_image_name(name: object) -> bool:
    if not isinstance(name, str) or not name or name != name.strip():
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return False
    try:
        _reject_windows_path_hazards((name,), role="image name")
    except BDDPreparationError:
        return False
    return name.endswith(".jpg")


def _image_members(
    members: Sequence[tuple[zipfile.ZipInfo, str, bool]], *, expected_image_count: int
) -> list[tuple[zipfile.ZipInfo, str]]:
    """Require the val-only BDD100K image layout and expected image cardinality."""

    regular = [(info, path) for info, path, is_directory in members if not is_directory]
    if not regular:
        raise BDDPreparationError("image archive contains no image files")
    # The official package is usually a full 100K archive.  Ignore train/test
    # members, but validate every member's path/link metadata first; only the
    # validation prefix is extracted below.
    prefix_matches = [
        prefix
        for prefix in IMAGE_LAYOUT_PREFIXES
        if any(path.startswith(f"{prefix}/") for _, path in regular)
    ]
    if len(prefix_matches) != 1:
        raise BDDPreparationError("image archive does not have the official 100k/val layout")
    selected: list[tuple[zipfile.ZipInfo, str]] = []
    names: set[str] = set()
    prefix = prefix_matches[0]
    for info, path in regular:
        if not path.startswith(f"{prefix}/"):
            continue
        relative = path.removeprefix(f"{prefix}/")
        if "/" in relative or not _is_safe_image_name(relative):
            raise BDDPreparationError("image archive contains a non-canonical validation image")
        key = relative.casefold()
        if key in names:
            raise BDDPreparationError("image archive contains duplicate image names")
        names.add(key)
        selected.append((info, relative))
    if len(selected) != expected_image_count:
        raise BDDPreparationError(
            f"BDD100K val image count must be {expected_image_count}, got {len(selected)}"
        )
    return sorted(selected, key=lambda item: item[1])


def _label_member(members: Sequence[tuple[zipfile.ZipInfo, str, bool]]) -> zipfile.ZipInfo:
    """Find the one official det_20 val JSON payload in a prevalidated ZIP."""

    matches = [
        info
        for info, path, is_directory in members
        if not is_directory and path in LABEL_MEMBER_PATHS
    ]
    if len(matches) != 1:
        raise BDDPreparationError("labels archive must contain exactly one det_20/det_val.json")
    return matches[0]


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise BDDPreparationError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise BDDPreparationError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise BDDPreparationError(f"{field} must be a finite number")
    return number


def _validate_det_val_frames(
    value: object, *, image_names: Iterable[str], expected_image_count: int
) -> dict[str, Any]:
    """Validate BDD det_20 frame identity, category ontology, and 2D boxes."""

    if not isinstance(value, list):
        raise BDDPreparationError("det_val JSON must be an array of frames")
    if len(value) != expected_image_count:
        raise BDDPreparationError(
            f"det_val JSON frame count must be {expected_image_count}, got {len(value)}"
        )
    expected_names = set(image_names)
    if len(expected_names) != expected_image_count:
        raise BDDPreparationError("image manifest contains non-unique image names")
    seen_names: set[str] = set()
    observed_categories: set[str] = set()
    annotation_count = 0
    allowed_categories = set(BDD100K_DETECTION_CATEGORIES)
    for frame_index, frame in enumerate(value):
        if not isinstance(frame, Mapping):
            raise BDDPreparationError(f"det_val frame {frame_index} must be an object")
        name = frame.get("name")
        if not _is_safe_image_name(name):
            raise BDDPreparationError(f"det_val frame {frame_index} has an unsafe image name")
        assert isinstance(name, str)  # Narrowed by _is_safe_image_name.
        if name in seen_names:
            raise BDDPreparationError("det_val JSON contains duplicate image names")
        seen_names.add(name)
        labels = frame.get("labels")
        if not isinstance(labels, list):
            raise BDDPreparationError(f"det_val frame {name} has no labels array")
        for label_index, label in enumerate(labels):
            if not isinstance(label, Mapping):
                raise BDDPreparationError(f"det_val frame {name} has a non-object label")
            category = label.get("category")
            if category not in allowed_categories:
                raise BDDPreparationError(
                    f"det_val frame {name} label {label_index} has a non-BDD100K category"
                )
            assert isinstance(category, str)  # Checked by membership in a str set.
            box = label.get("box2d")
            if not isinstance(box, Mapping):
                raise BDDPreparationError(
                    f"det_val frame {name} label {label_index} has no box2d object"
                )
            x1 = _finite_number(box.get("x1"), field="box2d.x1")
            y1 = _finite_number(box.get("y1"), field="box2d.y1")
            x2 = _finite_number(box.get("x2"), field="box2d.x2")
            y2 = _finite_number(box.get("y2"), field="box2d.y2")
            if x2 <= x1 or y2 <= y1:
                raise BDDPreparationError(f"det_val frame {name} has an empty box2d")
            observed_categories.add(category)
            annotation_count += 1
    if seen_names != expected_names:
        missing = len(expected_names - seen_names)
        unexpected = len(seen_names - expected_names)
        raise BDDPreparationError(
            "det_val image names must exactly match validation images "
            f"(missing={missing}, unexpected={unexpected})"
        )
    # A validation split (and, in particular, a reduced fixture used for a
    # smoke test) is not required to contain an instance of every ontology
    # category.  The official evaluator handles absent categories naturally;
    # rejecting such a split here would make preparation depend on incidental
    # class frequency rather than on the ontology contract.  Unknown labels
    # are still rejected above, and the public inventory continues to expose
    # the complete official category order.
    return {
        "frame_count": len(value),
        "annotation_count": annotation_count,
        "categories": list(BDD100K_DETECTION_CATEGORIES),
    }


def _archive_receipt(
    *,
    role: str,
    filename: str,
    sha256: str,
    byte_count: int,
    package_format: str,
    official_source_url: str,
    official_package_md5: str | None,
) -> dict[str, object]:
    """Create a receipt record deliberately free of local filesystem paths."""

    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
    ):
        raise BDDPreparationError("source package filename is not safe for a receipt")

    record: dict[str, object] = {
        "role": role,
        # ``name`` is the portable inventory spelling; ``filename`` remains
        # as an explicit human-readable alias for older local tooling.
        "name": filename,
        "filename": filename,
        "format": package_format,
        "sha256": sha256,
        "bytes": byte_count,
        "official_source_url": official_source_url,
    }
    if official_package_md5 is not None:
        record["official_package_md5"] = official_package_md5
    return record


def _assert_source_unchanged(
    source: Path, expected: tuple[str, int, str | None], *, role: str
) -> None:
    actual = _hash_file(source, include_md5=expected[2] is not None)
    if actual != expected:
        raise BDDPreparationError(f"{role} changed while preparation was running")


def _prepare_bundle(
    *,
    images_zip: Path,
    labels_input: Path,
    data_root: Path,
    expected_image_count: int = DEFAULT_EXPECTED_IMAGE_COUNT,
    images_official_md5: str | None = None,
    labels_official_md5: str | None = None,
) -> dict[str, object]:
    """Prepare a local bundle from already-downloaded BDD100K package files.

    This is the testable implementation boundary.  It intentionally accepts a
    smaller ``expected_image_count`` for synthetic test archives; the CLI
    defaults to BDD100K's official 10,000-image validation split.
    """

    if isinstance(expected_image_count, bool) or not isinstance(expected_image_count, int):
        raise BDDPreparationError("expected image count must be a positive integer")
    if not 1 <= expected_image_count <= 100_000:
        raise BDDPreparationError("expected image count must be in [1, 100000]")
    images_input = images_zip.expanduser()
    labels_input = labels_input.expanduser()
    data_input = data_root.expanduser()
    if images_input.is_symlink() or labels_input.is_symlink():
        raise BDDPreparationError("images ZIP and labels input must not be symbolic links")
    if data_input.is_symlink():
        raise BDDPreparationError("data root must not be a symbolic link")
    images_zip = images_input.resolve()
    labels_input = labels_input.resolve()
    data_root = data_input.resolve()
    if not images_zip.is_file() or not labels_input.is_file():
        raise BDDPreparationError("images ZIP and labels input must be existing local files")
    if images_zip.suffix.lower() != ".zip":
        raise BDDPreparationError("--images-zip must name a ZIP archive")
    if data_root.exists():
        raise BDDPreparationError(
            "data root must not already exist; choose a new ignored directory"
        )

    images_md5 = _normalise_expected_md5(images_official_md5, option="--images-md5")
    labels_md5 = _normalise_expected_md5(labels_official_md5, option="--labels-md5")
    images_source = _hash_file(images_zip, include_md5=images_md5 is not None)
    labels_source = _hash_file(labels_input, include_md5=labels_md5 is not None)
    _verify_official_md5(images_source[2], images_md5, role="images package")
    _verify_official_md5(labels_source[2], labels_md5, role="labels package")

    try:
        data_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{data_root.name}.prepare-", dir=data_root.parent))
    except OSError as exc:
        raise BDDPreparationError("cannot create the requested local data root") from exc
    moved = False
    try:
        output_images = staging / "images" / SPLIT
        output_labels = staging / "labels" / "det_val.json"
        with zipfile.ZipFile(images_zip) as archive:
            image_members = _image_members(
                _validated_zip_members(archive), expected_image_count=expected_image_count
            )
            images: list[dict[str, object]] = []
            for info, name in image_members:
                digest, byte_count = _safe_extract_member(archive, info, output_images / name)
                images.append({"name": name, "sha256": digest, "bytes": byte_count})

        labels_is_zip = labels_input.suffix.lower() == ".zip"
        if labels_is_zip:
            with zipfile.ZipFile(labels_input) as archive:
                info = _label_member(_validated_zip_members(archive))
                labels_sha256, labels_bytes = _safe_extract_member(archive, info, output_labels)
        else:
            labels_sha256, labels_bytes = _safe_copy_file(labels_input, output_labels)
        images.sort(key=lambda item: str(item["name"]))
        image_tree_sha256 = canonical_sha256(images)

        try:
            labels_payload = load_strict_json(output_labels)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise BDDPreparationError("det_val JSON must be valid strict UTF-8 JSON") from exc
        label_frames = _validate_det_val_frames(
            labels_payload,
            image_names=(str(item["name"]) for item in images),
            expected_image_count=expected_image_count,
        )
        _assert_source_unchanged(images_zip, images_source, role="images package")
        _assert_source_unchanged(labels_input, labels_source, role="labels package")

        source_archives = [
            _archive_receipt(
                role="images_val_zip",
                filename=images_zip.name,
                sha256=images_source[0],
                byte_count=images_source[1],
                package_format="zip",
                official_source_url=BDD100K_IMAGES_VAL_URL,
                official_package_md5=images_md5,
            ),
            _archive_receipt(
                role="det_20_labels",
                filename=labels_input.name,
                sha256=labels_source[0],
                byte_count=labels_source[1],
                package_format="zip" if labels_is_zip else "json",
                official_source_url=BDD100K_DET20_VAL_LABELS_URL,
                official_package_md5=labels_md5,
            ),
        ]
        image_manifest: dict[str, object] = {
            "schema_version": IMAGE_MANIFEST_SCHEMA,
            "dataset_id": DATASET_ID,
            "task": TASK,
            "split": SPLIT,
            "image_count": expected_image_count,
            "images_tree_sha256": image_tree_sha256,
            "images": images,
            "source_archives": source_archives,
        }
        split_inventory: dict[str, object] = {
            "schema_version": SPLIT_INVENTORY_SCHEMA,
            "dataset_id": DATASET_ID,
            "task": TASK,
            "split": SPLIT,
            "image_directory": "images/val",
            "image_manifest": "image-manifest.json",
            "image_count": expected_image_count,
            "images_tree_sha256": image_tree_sha256,
            "labels": {
                "path": "labels/det_val.json",
                "sha256": labels_sha256,
                "bytes": labels_bytes,
                **label_frames,
            },
        }
        content_sha256 = canonical_sha256(
            {
                "dataset_id": DATASET_ID,
                "task": TASK,
                "split": SPLIT,
                "images_tree_sha256": image_tree_sha256,
                "labels_sha256": labels_sha256,
                "image_count": expected_image_count,
            }
        )
        dataset_manifest = DatasetManifest.model_validate(
            {
                "schema_version": "roadsense.dataset-manifest/v1",
                "dataset_name": "BDD100K Detection 2020 validation",
                "source_url": BDD100K_SOURCE_URL,
                "license_id": "BDD100K-research-license-accepted-locally",
                "tasks": [TASK],
                "splits": {
                    SPLIT: (
                        f"{expected_image_count} images; det_20/det_val.json; "
                        "local research-license preparation"
                    )
                },
                "content_sha256": content_sha256,
                "evaluation_authorized": True,
                "frozen": False,
                "notes": (
                    "Local-only BDD100K Detection 2020 validation preparation. The operator "
                    "explicitly accepted the BDD100K research license at preparation time. "
                    f"Image tree SHA-256={image_tree_sha256}; det_val SHA-256={labels_sha256}."
                ),
            }
        )
        source_receipt: dict[str, object] = {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "dataset_id": DATASET_ID,
            "task": TASK,
            "split": SPLIT,
            "local_only": True,
            "license_acceptance": {
                "accepted": True,
                "flag": "--accept-bdd100k-research-license",
                "scope": "BDD100K research license accepted by the local operator",
            },
            "source_archives": source_archives,
            "prepared_layout": {
                "images": "images/val",
                "labels": "labels/det_val.json",
                "image_manifest": "image-manifest.json",
                "split_inventory": "split-inventory.json",
                "dataset_manifest": "dataset-manifest.json",
            },
            "images_tree_sha256": image_tree_sha256,
            "labels_sha256": labels_sha256,
            "labels_bytes": labels_bytes,
            "content_sha256": content_sha256,
        }
        write_json_atomic(staging / "image-manifest.json", image_manifest)
        write_json_atomic(staging / "split-inventory.json", split_inventory)
        write_json_atomic(
            staging / "dataset-manifest.json", dataset_manifest.model_dump(mode="json")
        )
        write_json_atomic(staging / "source-receipt.json", source_receipt)

        # Do not overwrite an output created by another process after the
        # initial check.  ``rename`` is a same-directory commit on the normal
        # local filesystem and does not deliberately replace a user bundle.
        if data_root.exists():
            raise BDDPreparationError(
                "data root appeared during preparation; refusing to overwrite"
            )
        staging.rename(data_root)
        moved = True
        return {
            "image_manifest": "image-manifest.json",
            "split_inventory": "split-inventory.json",
            "dataset_manifest": "dataset-manifest.json",
            "source_receipt": "source-receipt.json",
            "image_count": expected_image_count,
            "images_tree_sha256": image_tree_sha256,
            "content_sha256": content_sha256,
        }
    except (OSError, zipfile.BadZipFile, UnicodeError, RecursionError) as exc:
        raise BDDPreparationError("BDD100K local preparation failed") from exc
    finally:
        if not moved:
            shutil.rmtree(staging, ignore_errors=True)


def prepare_bdd100k_detection(
    *,
    images_zip: Path,
    labels_input: Path,
    data_root: Path,
    expected_image_count: int = DEFAULT_EXPECTED_IMAGE_COUNT,
    images_official_md5: str | None = None,
    labels_official_md5: str | None = None,
    accept_bdd100k_research_license: bool = False,
) -> dict[str, object]:
    """Public preparation boundary requiring an explicit local licence acknowledgement."""

    if accept_bdd100k_research_license is not True:
        raise BDDPreparationError(
            "pass --accept-bdd100k-research-license after reviewing the BDD100K research license"
        )
    return _prepare_bundle(
        images_zip=images_zip,
        labels_input=labels_input,
        data_root=data_root,
        expected_image_count=expected_image_count,
        images_official_md5=images_official_md5,
        labels_official_md5=labels_official_md5,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-zip",
        "--images-val-zip",
        "--images-val",
        "--images",
        dest="images_zip",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--labels",
        "--labels-zip",
        "--labels-val-zip",
        "--labels-file",
        dest="labels_input",
        type=Path,
        required=True,
        help="Official det_20 labels ZIP, or a local det_val.json file already extracted from it.",
    )
    parser.add_argument(
        "--data-root",
        "--output-root",
        type=Path,
        default=Path("data/raw/bdd100k_detection_2020_val"),
        help="New ignored local data directory. Existing directories are never overwritten.",
    )
    parser.add_argument(
        "--accept-bdd100k-research-license",
        action="store_true",
        help="Required acknowledgement that the local operator reviewed and accepts the BDD100K research license.",
    )
    parser.add_argument(
        "--expected-image-count",
        "--expected-images",
        type=int,
        default=DEFAULT_EXPECTED_IMAGE_COUNT,
        help="Validation image count; defaults to the official BDD100K val count (10000).",
    )
    parser.add_argument(
        "--images-md5",
        "--images-package-md5",
        "--official-images-md5",
        dest="images_md5",
        help="Optional official MD5 for the supplied images package; checked before extraction.",
    )
    parser.add_argument(
        "--labels-md5",
        "--labels-package-md5",
        "--official-labels-md5",
        dest="labels_md5",
        help="Optional official MD5 for the supplied det_20 labels package; checked before extraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.  It reports only relative artifact names, never input paths."""

    args = _build_parser().parse_args(argv)
    try:
        result = prepare_bdd100k_detection(
            images_zip=args.images_zip,
            labels_input=args.labels_input,
            data_root=args.data_root,
            expected_image_count=args.expected_image_count,
            images_official_md5=args.images_md5,
            labels_official_md5=args.labels_md5,
            accept_bdd100k_research_license=args.accept_bdd100k_research_license,
        )
    except (BDDPreparationError, MemoryError) as exc:
        print(f"BDD100K preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
