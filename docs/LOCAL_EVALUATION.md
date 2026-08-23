# Local evaluation (operator-provided data only)

`roadsense evaluate-local` is the explicit real-data entry point.  It reads
only files named by a local `roadsense.local-evaluation/v1` specification.  It
does not fetch URLs, install model weights, or download a dataset.  A missing
or malformed input stops the run before a report is emitted.

## Dry-run the fixture

Use this path to verify the command and report plumbing without any dataset:

```powershell
.\.venv\Scripts\python.exe -m roadsense evaluate-local --fixture --json
```

The output deliberately contains `evidence_level=fixture`,
`evaluation_authorized=false`, `frozen=false`, and
`benchmark_claim_available=false`.  The deterministic values are not real
model or public-dataset metrics.

## Input layout

An evaluation spec is a strict JSON object.  Relative paths are resolved from
the spec's directory; absolute local paths are also allowed.  `http://`,
`https://`, cloud URIs, and other URL schemes are rejected.

`configs/local_evaluation_template.json` is a non-data template with the same
shape. Replace every path with operator-provided licensed files before use;
the template is not a runnable benchmark fixture.

```json
{
  "schema_version": "roadsense.local-evaluation/v1",
  "dataset_manifest": "dataset-manifest.json",
  "split": "development",
  "split_sequences": {
    "development": ["video-0001", "video-0002"],
    "final": ["video-0003"]
  },
  "tasks": ["detection", "segmentation", "tracking"],
  "ground_truth": "ground-truth.json",
  "predictions": "predictions.json",
  "model_artifact": {
    "manifest": "artifacts/detector-manifest.json",
    "root": "artifacts"
  },
  "segmentation": {
    "ground_truth": {
      "video-0001": "labels/video-0001.npy",
      "video-0002": "labels/video-0002.npy"
    },
    "predictions": {
      "video-0001": "outputs/video-0001.npy",
      "video-0002": "outputs/video-0002.npy"
    },
    "num_classes": 5,
    "ignore_index": 255
  },
  "protocols": {
    "detection_iou_threshold": 0.5,
    "tracking_iou_threshold": 0.5
  }
}
```

`model_artifact` is optional for exploratory development runs.  When present,
the evaluator loads the local manifest, verifies the checkpoint and dependency
lock SHA-256 below the supplied root, and records the immutable verification
receipt in `details.model_artifact` and `details.input_hashes`.  It never loads
the model runtime.  A dataset manifest marked `evaluation_authorized=true` or
`frozen=true` must include this block; otherwise the run fails closed instead
of producing an authorized report without model provenance.

`split_sequences` is required, and every sequence ID may occur in only one
split.  The selected split must also be declared by the accompanying
`roadsense.dataset-manifest/v1`.  Ground-truth and prediction bundles must
contain exactly the selected IDs; this prevents accidentally evaluating a
frame-level mixture of train and final data.

Each bundle is a strict JSON object:

```json
{
  "schema_version": "roadsense.sequence-bundle/v1",
  "sequences": [
    {
      "sequence_id": "video-0001",
      "frames": [
        {
          "frame_index": 0,
          "timestamp_ms": 0,
          "image_size": {"width": 1280, "height": 720},
          "detections": [
            {
              "category_id": 1,
              "score": 0.91,
              "track_id": 7,
              "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 160}
            }
          ]
        }
      ]
    }
  ]
}
```

Frame indices must be strictly increasing, timestamps cannot regress, and
truth/prediction frame indices, timestamps, and image sizes must align.  A
segmentation `.npy` file must be an integer array with shape
`(frame_count, height, width)` and is loaded with `allow_pickle=False`.
The raster height/width are the frozen segmentation coordinate space and may
be lower-resolution than the frame's `image_size`; ground truth and
predictions must use the same raster shape for every selected sequence.

## Running and interpreting a report

```powershell
.\.venv\Scripts\python.exe -m roadsense evaluate-local .\data\local\evaluation.json `
  --output .\reports\local-development.json --json
```

The report includes per-task aggregate and per-sequence diagnostics, split
IDs, manifest/spec/input SHA-256 hashes, and a short `report_id` bound to the
complete payload.  The report explicitly says whether a verified model
artifact is bound; without one it binds prediction-file provenance only.
`dataset_manifest_sha256` is the canonical hash of the manifest JSON itself;
the upstream/archive identity is reported separately as
`details.dataset_content_sha256`.
Unless the manifest explicitly sets both
`evaluation_authorized=true` and `frozen=true`, the report remains development
evidence and cannot pass the publication gate.  Compact RoadSense AP/mIoU/
MOTA/identity-F1 values must not be relabeled as official COCO, BDD100K,
TrackEval, HOTA, or IDF1 results.

Raw images, videos, labels, checkpoints, and private paths are local inputs;
they are not copied into Git, Pages, source distributions, or reports.
