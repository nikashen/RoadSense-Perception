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
& $py scripts\run_bdd100k_detection_benchmark.py evaluate `
  --ground-truth data\raw\bdd100k_detection_2020_val\labels\det_val.json `
  --predictions runs\bdd100k_val\inference\predictions.json `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --evaluator-python D:\bdd-eval-venv\Scripts\python.exe `
  --evaluator-cwd D:\bdd100k-devkit-9ac17c6c `
  --output-dir runs\bdd100k_val\evaluate-a

& $py scripts\run_bdd100k_detection_benchmark.py evaluate `
  --ground-truth data\raw\bdd100k_detection_2020_val\labels\det_val.json `
  --predictions runs\bdd100k_val\inference\predictions.json `
  --image-manifest runs\bdd100k_val\frozen-image-manifest.json `
  --evaluator-python D:\bdd-eval-venv\Scripts\python.exe `
  --evaluator-cwd D:\bdd100k-devkit-9ac17c6c `
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
