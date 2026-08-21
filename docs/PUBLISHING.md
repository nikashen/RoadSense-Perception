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

Raw BDD100K images, videos, labels, private weights, and local paths never go
into GitHub, Pages, or release assets by default. The publication gate must
remain fail-closed when authorization or provenance is incomplete.
