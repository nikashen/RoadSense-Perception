# Publishing Checklist

This project has two publication modes.

## Development snapshot

Allowed before real evaluation: source code, tests, deterministic Pages
fixture, wheel/sdist, architecture and protocol documents, and explicit
`fixture` evidence flags. The README must not contain real-data metric claims.

## Benchmark release

Do not create a benchmark release until all of the following are present:

1. Current dataset terms reviewed and recorded.
2. For the BDD lane, the official 10,000-image Detection 2020 `val` bundle
   passed `scripts/prepare_bdd100k_detection.py` with the explicit licence
   acknowledgement and the published Berkeley MD5 values for both ZIPs.
   A Kaggle/community `det_val.json` or a raw mirror URL without matching
   checksums is not acceptable evidence.
3. Archive/tree/ground-truth hashes and a frozen image manifest are bound to
   the model, ontology adapter, inference configuration, and predictions.
4. Two independent runs of the pinned official BDD devkit completed with
   finite metrics, distinct run IDs, byte-for-byte identical aggregate
   metrics, and the complete validated evaluator lock documented in the BDD
   benchmark guide (including `pycocotools==2.0.7`). See
   [BDD100K benchmark](BDD100K_BENCHMARK.md).
5. A sanitized receipt passes
   `scripts/verify_bdd100k_detection_benchmark.py`; it contains no local
   paths, raw predictions, labels, images, or private weights.
6. Reproducible runtime measurements with device and dependency versions. The
   fixture `roadsense.runtime-audit/v1` record described in
   [Runtime audit](RUNTIME_AUDIT.md) is a plumbing diagnostic only; it cannot
   substitute for a real-model record.
7. Release assets pass `twine check`, fresh-wheel smoke, and Pages verification.

The Pages artifact contains a generated `demo.json` from the same deterministic
payload as `/api/v1/demo`; `scripts/verify_pages.py` checks its canonical hash,
asset hashes, relative HTML references, and fixture-only evidence flags.

For a local release-candidate check, install the release checker explicitly
and validate both archives before uploading anything:

```powershell
.\.venv\Scripts\python.exe -m pip install "twine>=5,<7"
.\.venv\Scripts\python.exe -m build --sdist --wheel --outdir dist
$artifacts = Get-ChildItem -LiteralPath dist -File |
  Where-Object { $_.Name -match '\.(whl|tar\.gz)$' }
if ($artifacts.Count -lt 2) { throw "wheel and sdist are required" }
& .\.venv\Scripts\python.exe -m twine check $artifacts.FullName
```

Then install the wheel (and, separately, the sdist) into a fresh Python
environment and run `roadsense smoke --json` plus the local API asset check.
The CI workflow performs the wheel check on Linux and Windows; no release is
authorized by this checklist alone.

Raw BDD100K images, videos, labels, private weights, and local paths never go
into GitHub, Pages, or release assets by default. The publication gate must
remain fail-closed when authorization or provenance is incomplete.

The current repository has the BDD runner and offline contract tests, but no
public BDD benchmark receipt until an operator supplies the licensed official
data and the two evaluator runs. A synthetic fixture or COCO8 result is never
promoted to a BDD claim.
