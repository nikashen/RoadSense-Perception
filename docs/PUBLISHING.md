# Publishing Checklist

This project has two publication modes.

## Development snapshot

Allowed before real evaluation: source code, tests, deterministic Pages
fixture, wheel/sdist, architecture and protocol documents, and explicit
`fixture` evidence flags. The README must not contain real-data metric claims.

## Benchmark release

Do not create a benchmark release until all of the following are present:

1. Current dataset terms reviewed and recorded.
2. Archive/tree hashes and sequence-aware split manifest.
3. Model artifact manifest with checkpoint hash and license.
4. Frozen Development configuration and independent Final evaluator receipt.
5. Reproducible runtime measurements with device and dependency versions.
6. Sanitized aggregate report whose hashes bind the inputs above.
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
