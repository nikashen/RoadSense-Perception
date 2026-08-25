# BDD100K Detection benchmark

This repository defines one deliberately narrow formal benchmark lane:

`BDD100K Detection 2020 validation` (`val`), evaluated by the official
`bdd100k` devkit at commit
`9ac17c6c7c51d2fc83065fccd707cd5b1882a293`.

It is a local validation benchmark, not a hidden-test leaderboard submission.
Tracking, semantic segmentation, and EvalAI/test results are separate future
lanes and must not be mixed into this receipt.

## Data and licence boundary

BDD100K is not downloaded by the repository and is not redistributed by it.
The operator must obtain the official packages from Berkeley, review the
current BDD100K research/education licence, and explicitly pass
`--accept-bdd100k-research-license` to the preparation command. The images,
labels, predictions, model files, and local paths stay outside GitHub, Pages,
and release assets.

The official package checksums documented by Berkeley for this lane are:

| package | official MD5 |
| --- | --- |
| `bdd100k_images_100k.zip` | `5a0359c86a0b8713adab1eee9a3041cb` |
| `bdd100k_det_20_labels.zip` | `b86a3e1b7edbcad421b7dad2b3987c94` |

The preparation script also records SHA-256 and validates the complete
10,000-image `images/100k/val` inventory. These values are provenance checks,
not permission to redistribute the dataset.

The Berkeley portal's historical raw mirror (`128.32.162.150`) is not a
content-addressed API and must not be treated as proof of identity. In
particular, a stale/misconfigured labels link has returned a different ZIP in
the past. The formal preparation command therefore requires both MD5 values
and a ZIP labels package; an extracted `det_val.json`, Kaggle export, or
community subset is development-only and is rejected for the 10,000-image
lane. The receipt records the Berkeley portal URL for provenance, while the
MD5 values—not the URL—identify the bytes.

During the August 2026 validation, the mirror's image ZIP was complete and
readable but hashed to `c7a1d4db9af5a4691d5f6fee2a62e132`, rather than the
published image MD5. The mirror's labels URL returned a 231,196,019-byte ZIP
of JPEG files with MD5 `e72531b982bbb42efbaaf93223527284`, not a labels
archive. The official labels MD5 remains documented in the pinned upstream
devkit documentation, but no reachable source in this run produced those
bytes. Consequently, no formal BDD metric or benchmark release is implied by
the development evaluator smoke records.

## Reproduction stages

Run each stage with the project environment. The paths below are examples and
should point to ignored local directories.

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py scripts\prepare_bdd100k_detection.py `
  --images-zip D:\bdd100k\bdd100k_images_100k.zip `
  --labels D:\bdd100k\bdd100k_det_20_labels.zip `
  --data-root data\raw\bdd100k_detection_2020_val `
  --images-md5 5a0359c86a0b8713adab1eee9a3041cb `
  --labels-md5 b86a3e1b7edbcad421b7dad2b3987c94 `
  --accept-bdd100k-research-license

& $py scripts\run_bdd100k_detection_benchmark.py freeze `
  --manifest data\raw\bdd100k_detection_2020_val\image-manifest.json `
  --output runs\bdd100k_val\frozen-image-manifest.json

& $py scripts\run_bdd100k_detection_benchmark.py infer `
  --model data\raw\real_eval\yolo11n.onnx `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --images-root data\raw\bdd100k_detection_2020_val\images\val `
  --output-dir runs\bdd100k_val\inference

# Run this command twice in independent output directories/environments.
# The evaluator environment must contain pycocotools==2.0.7.  The tested
# 2.0.9/2.0.10 releases are incompatible with the pinned bdd100k/scalabel
# path; the formal lane deliberately pins the known-good 2.0.7 build.
& $py scripts\run_bdd100k_detection_benchmark.py evaluate `
  --ground-truth data\raw\bdd100k_detection_2020_val\labels\det_val.json `
  --predictions runs\bdd100k_val\inference\predictions.json `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --evaluator-python D:\bdd-eval-venv\Scripts\python.exe `
  --evaluator-cwd D:\bdd100k-devkit-9ac17c6c `
  --role independent_a `
  --output-dir runs\bdd100k_val\evaluate-a

& $py scripts\run_bdd100k_detection_benchmark.py evaluate `
  --ground-truth data\raw\bdd100k_detection_2020_val\labels\det_val.json `
  --predictions runs\bdd100k_val\inference\predictions.json `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --evaluator-python D:\bdd-eval-venv\Scripts\python.exe `
  --evaluator-cwd D:\bdd100k-devkit-9ac17c6c `
  --role independent_b `
  --output-dir runs\bdd100k_val\evaluate-b

& $py scripts\finalize_bdd100k_detection_benchmark.py `
  --source-receipt data\raw\bdd100k_detection_2020_val\source-receipt.json `
  --dataset-manifest data\raw\bdd100k_detection_2020_val\dataset-manifest.json `
  --split-inventory data\raw\bdd100k_detection_2020_val\split-inventory.json `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --model-manifest runs\bdd100k_val\inference\model-manifest.json `
  --inference-receipt runs\bdd100k_val\inference\inference-receipt.json `
  --evaluation-a runs\bdd100k_val\evaluate-a\evaluation-receipt.json `
  --evaluation-b runs\bdd100k_val\evaluate-b\evaluation-receipt.json `
  --output benchmark-receipt.json

& $py scripts\verify_bdd100k_detection_benchmark.py benchmark-receipt.json
```

The inference process receives only the frozen image manifest and image
directory; it has no ground-truth argument. The evaluator is an external
Python process pinned to the devkit commit. Every stage writes hashes and
relative artifact names. The final receipt contains only aggregate finite
metrics and hash bindings.

For a formal 10,000-image run, evaluator receipts must record
`pycocotools==2.0.7`, `returncode=0`, `timed_out=false`, and a parsed result
file (stdout fallback is not publishable). Small synthetic evaluator fixtures
used by contract tests intentionally do not satisfy the formal publication
gate.

The evaluator environment is a separate, non-editable installation of the
official devkit source at the commit above. The source archive used during
development was SHA-256
`0b1b3b40cd17bb4d7c9be0a0014aac60fc007ca4d7855a0237a52042ec8c193a`; the
known-good runtime includes `bdd100k==1.0.0`, `scalabel==0.3.1`,
`numpy==1.26.4`, `pydantic==1.10.15`, `motmetrics==1.4.0`, and
`pycocotools==2.0.7`. These versions describe the tested evaluator runtime;
the virtual environment itself is local-only and must not be committed.

## Model interpretation

The checked-in runner currently uses an Ultralytics YOLO11n COCO ONNX artifact
as an explicitly labelled cross-domain baseline. Its ontology adapter maps
only faithful COCO classes (`person`, `bicycle`, `car`, `motorcycle`, `bus`,
`train`, `truck`, and `traffic light`). It never guesses `rider` or
`traffic sign`. A benchmark receipt therefore describes this exact baseline;
it must not be described as BDD-trained unless a separately documented,
hash-bound BDD training run and licence is supplied.

## Publication gate

Do not create a release or put `benchmark_claim_available=true` in public
artifacts until all of these are true:

1. the official 10,000-image val bundle passed preparation and licence checks;
2. the frozen image manifest, model manifest, inference receipt, GT hash, and
   prediction hash agree;
3. two independent official-devkit runs completed with `status=ok`, finite
   metrics, distinct run IDs, and identical aggregate metrics;
4. `scripts/verify_bdd100k_detection_benchmark.py` succeeds on the sanitized
   receipt; and
5. release assets contain no BDD media/labels/predictions, private weights,
   absolute paths, or credentials.

Before those conditions, the repository remains a development snapshot and
the Pages demo remains the deterministic fixture described in the main README.
