# Model Artifact Protocol

RoadSense never loads a model as a side effect of importing `roadsense`,
starting the fixture API, or opening the Pages demo. A real detector,
segmenter, or tracker adapter must receive an explicit
`roadsense.model-artifact-manifest/v1` document and pass local hash
verification before it is allowed into an inference runner.

## Required binding

`roadsense.adapters.ModelArtifactManifest` binds all values that can change a
reported result:

- a stable `artifact_id`, normalized repository-relative `artifact_path`,
  non-placeholder checkpoint SHA-256, and optional byte size;
- artifact format, framework/backend names and versions;
- task set and a unique numeric/label ontology;
- input dimensions, channels, dtype, layout, color order, and coordinate space;
- resize mode, scale, normalization mean/std, and padding value;
- runtime name/version, device, precision, determinism flag, and optional ONNX
  opset;
- license/source receipt and, when used, a dependency-lock path and SHA-256.

The manifest itself is canonical-JSON hashed. The resulting digest is included
in `ArtifactVerification`, so a report can bind both the checkpoint and the
metadata that describes how it was executed.

## Fail-closed verification

`verify_artifact_manifest(manifest, artifact_root=...)` performs only local
filesystem reads. It rejects missing files, size/hash mismatches, absolute or
`..` paths, symlink escapes outside the supplied root, malformed JSON,
duplicate keys, non-finite values, duplicate tasks/classes, and incomplete
dependency-lock pairs. It never follows a URL and never imports PyTorch,
ONNX Runtime, OpenCV, or another model framework.

The CLI exposes the same boundary:

```powershell
# Schema/metadata dry-run. No checkpoint is opened and model_loaded=false.
roadsense audit-artifact artifacts/detector-manifest.json --json

# Explicit local verification. Still does not instantiate a model.
roadsense verify-artifact artifacts/detector-manifest.json --root artifacts --output reports/artifact-receipt.json
```

Without `--root`, the command validates only the manifest. With `--root`, it
returns a receipt with `verified=true`, observed size/hash, and
`model_loaded=false`; an inference adapter must make a separate, explicit
runtime-loading decision.

## Adapter seam and registry

`ModelAdapter[Input, Output]` is a small protocol with three members:
`adapter_id`, `manifest`, and `infer(frame)`. `AdapterRegistry` validates the
identity/manifest pairing, rejects duplicate IDs, and can require a verified
receipt before inference. Registration without an artifact root is useful for
contract tests only and remains unverified; `require_verified()` rejects it.

The protocol deliberately does not prescribe Torch, ONNX, TensorRT, or a media
decoder. Optional integrations should translate their native outputs into the
frozen `FrameRecord`/metric contracts and record runtime/device details in the
evaluation report.

When a local evaluation is marked authorized or frozen, its spec must include
the manifest path and an allow-listed artifact root. `roadsense evaluate-local`
verifies that pair before reading predictions, checks that the declared task
set and ontology cover the run, and embeds the immutable receipt in the report.
Exploratory development runs may omit the block, but the report then states
explicitly that only prediction-file provenance is bound.

Weights and private training runs are local inputs. They are excluded from Git,
source distributions, Pages, and releases unless their license explicitly
allows redistribution and a separate publication review approves them.
